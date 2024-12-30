#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import random
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from sklearn.metrics import roc_auc_score
import sys
import pickle
from rank_bm25 import BM25Okapi
from entselect_model import ExtractorModel
from entselect_dataset import DatasetExtractor
sys.path.append("../")
from NameEntityRecognition.ner_infere import NERInfere
from utils import transform_attrname2inputname, transform_inputname2attrname,load_kb, BM25_Macth
from utils import seed_everything,attrs2str, get_attrname2entities


# In[2]:


### metric
def hit_1(y, x):
    idx = x.argmax()
    if y[idx]==1:
        return 1
    return 0

def gorup_metric_fn(df):
    if len(np.unique(df['labels']))==2:
        df['auc'] = roc_auc_score(df['labels'],df['logits'])
    else:
#         print("all label same:",df['labels'])
        df['auc']=0.5
    
    df['hit_1'] =  hit_1(df['labels'].values, df['logits'].values)
    return df

def cal_acc_score(ids, logits, labels):
    df = pd.DataFrame({'ids':ids.squeeze(-1).tolist(), 'logits':logits.squeeze(-1).tolist(), 'labels':labels.squeeze(-1).tolist()})
    df1 = df.groupby('ids',as_index=False,sort=True).apply(gorup_metric_fn)
    df1 = df1[['ids','auc','hit_1']].drop_duplicates(ignore_index=True)
    auc = df1['auc'].mean()
    acc = df1['hit_1'].mean()
    return (auc, acc)


# In[3]:


def train(args):
    pretrain_model_path = args.pretrain_model_path
    save_model_path = args.save_model_path
    os.makedirs(save_model_path, exist_ok=True)
    
    gpu = args.gpu
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs
    
    device = torch.device(gpu)
    print("Loading dataset...")

    with open(args.train_file, 'r',  encoding='utf-8') as f:
        train_data = json.load(f)
    with open(args.dev_file, 'r',  encoding='utf-8') as f:
        dev_data = json.load(f)
    
    train_dataset = DatasetExtractor(train_data, args.max_conv_seq_len, args.max_entity_len, args.max_attrname_len,
                                     args.max_attrvalue_len, args.pretrain_model_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)   # 可以改为 shuffle=True
    
    
    print('Creating model...')
    model = ExtractorModel(device=device, model_path=args.pretrain_model_path)
    print('Model created!')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    optimizer.zero_grad()

    best_score = -float("inf")
    not_up_epoch = 0
    
    model.zero_grad()
    
    for epoch in range(nb_epochs):
        model.train()
        loader = tqdm(train_dataloader, total=len(train_dataloader),
                      unit="batches")
        running_loss = 0
        
        all_ids, all_ent_logits, all_ent_labels,all_attr_logits, all_attr_labels= [],[],[],[],[]
        for i_batch, data in enumerate(loader):
            model.zero_grad()
            text_id, inputs, ent_label, attr_label = data
            token_ids = inputs["input_ids"].flatten(1).to(device)
            attention_mask = inputs["attention_mask"].flatten(1).to(
                device)
            token_type_ids = inputs["token_type_ids"].flatten(1).to(
                device)
            ent_label = ent_label.to(device)
            
            ent_logit = model(
                token_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            ent_loss = loss_fn(ent_logit, ent_label)
            loss = ent_loss
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(
                Loss=running_loss / ((i_batch + 1) * batch_size),
                Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
            
            all_ids.append(text_id.detach().cpu().numpy())
            all_ent_logits.append(ent_logit.detach().cpu().numpy())
            all_ent_labels.append(ent_label.detach().cpu().numpy())

        all_ids = np.concatenate(all_ids,)
        all_ent_logits = np.concatenate(all_ent_logits,)
        all_ent_labels = np.concatenate(all_ent_labels,)
        trn_ent_auc, trn_ent_acc =  cal_acc_score(all_ids, all_ent_logits, all_ent_labels)
        
        print("train step %d ent: auc=%.6f, acc=%.6f"%(epoch+1, trn_ent_auc, trn_ent_acc))
        
        val_ent_auc, val_ent_acc  = validate(device, model, dev_data, args)
        print("valid step %d ent: auc=%.6f, acc=%.6f"%(epoch+1, val_ent_auc, val_ent_acc))
        score = val_ent_acc
        
        if epoch==nb_epochs-1:
            print("save final model for test...")
            torch.save(model.state_dict(),
                       os.path.join(save_model_path, "x.pt"))
        if score > best_score + 0.0001:
            best_score = score
            not_up_epoch = 0
            print(
                'Validation accuracy %f increased from previous epoch, '
                'save best_model' % score)
            torch.save(model.state_dict(),
                       os.path.join(save_model_path, "best_model.pt"))
        else:
            not_up_epoch += 1
            if not_up_epoch > 100:
                print(
                    f"Corrcoef didn't up for %s batch, early stop!"
                    % not_up_epoch)
                break


# In[4]:


def validate(device, model, dev_data, args):
    model.eval()
    
    dataset = DatasetExtractor(dev_data, args.max_conv_seq_len,  args.max_entity_len, args.max_attrname_len, args.max_attrvalue_len,
                                     args.pretrain_model_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # all_ids, all_logits, all_labels= [],[],[]
    all_ids, all_ent_logits, all_ent_labels,all_attr_logits, all_attr_labels= [],[],[],[],[]
    for i_batch, data in enumerate(tqdm(dataloader)):
        model.zero_grad()
        text_id, inputs, ent_label, attr_label = data
        token_ids = inputs["input_ids"].flatten(1).to(device)
        attention_mask = inputs["attention_mask"].flatten(1).to(
            device)
        token_type_ids = inputs["token_type_ids"].flatten(1).to(
            device)
        ent_label = ent_label.to(device)

        ent_logit = model(
            token_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        all_ids.append(text_id.detach().cpu().numpy())
        all_ent_logits.append(ent_logit.detach().cpu().numpy())
        all_ent_labels.append(ent_label.detach().cpu().numpy())

        
    all_ids = np.concatenate(all_ids,)
    all_ent_logits = np.concatenate(all_ent_logits,)
    all_ent_labels = np.concatenate(all_ent_labels,)
    ent_auc, ent_acc = cal_acc_score(all_ids, all_ent_logits, all_ent_labels)
    
    return (ent_auc, ent_acc)


# In[5]:


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--train_file', type=str, default='data/train_data.json')
parser.add_argument('--dev_file', type=str, default='data/val_data.json')
# parser.add_argument('--kb_file', type=str, default='../../data/kg.json')
parser.add_argument('--pretrain_model_path', type=str, default='../../pretrain_model/ernie-3.0-base-zh/')
parser.add_argument('--save_model_path', type=str, default='../model/entity_select_ernie3')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_conv_seq_len', type=int, default=400)
# parser.add_argument('--max_seq_len', type=int, default=64)
parser.add_argument('--max_entity_len', type=int, default=10)
parser.add_argument('--max_attrname_len', type=int, default=40) # 多实体
parser.add_argument('--max_attrvalue_len', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=5) # 10
parser.add_argument('--validate_every', type=int, default=1)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--neg_cnt', type=int, default=4)


parser.add_argument('--fp16', type=bool, default=False)
parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html", )

# sys.argv = ['']
args = parser.parse_args()

seed = 1
seed_everything(seed)


if args.mode == "train":
    train(args)
elif args.mode == "dev":
    pass
else:
    pass


# In[6]:


# train step 1 ent: auc=0.950302, acc=0.772228
# 100%|██████████| 696/696 [01:04<00:00, 10.77it/s]
# valid step 1 ent: auc=0.974261, acc=0.848152
# Validation accuracy 0.848152 increased from previous epoch, save best_model


# In[7]:


# train step 2 auc=0.964714, acc=0.888817
# valid step 2 auc=0.954600, acc=0.874711


# In[8]:


# length of data: 19976
# length of sample: 153389
# length of data: 4757
# length of sample: 35681

