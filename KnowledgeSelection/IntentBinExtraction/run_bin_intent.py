#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse

import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from sklearn.metrics import roc_auc_score
from extractor_model import ExtractorModel
from extractor_dataset import DatasetExtractor
import sys
sys.path.append('../')
from NameEntityRecognition.ner_infere import NERInfere
from utils import transform_attrname2inputname, transform_inputname2attrname,seed_everything,BCEFocalLoss
from utils import load_kb

def load_data(datafile, kb):
    data = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(json.loads(line.strip()))
    print(f"length of data: {len(data)}")

    samples = []
    for text_id,sample in enumerate(data):
#         if(text_id>1000):
#             break
        query = sample.get("question")  
        # {question: str，answer: str，knowledge: list(dict), context: list(str)，prev_entities: list(str)}
        entity2attr = {}     # answer中使用到的 entity to set(attrname)
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
#             if attrname == "Information":
#                 attrname = "简介"
            attrname = transform_attrname2inputname(attrname)
            if entity not in entity2attr:
                entity2attr[entity] = set()
            entity2attr.get(entity).add(attrname)
        
        entities = [entity for entity, attrs in entity2attr.items()]
        entities = sorted(entities)
        for entity in entities:
            attrs = entity2attr[entity]
            attrs = sorted(list(attrs))
            subgraph = kb.get(entity, {})  # entity 对于的所有attrname
            text1 = query.replace(entity, "ne")   # 不理解替换的意思
            for attr in attrs:
                text2 = attr
                attrvalue =  kb.get(entity)[text2]
                attrvalue = ','.join(attrvalue)
                samples.append([text_id, text1, text2, attrvalue, 1])       # (id, question, attrname, attrvalue)正样本
            for key in subgraph:  # 这里加入顺序
                if key not in attrs:    # 优化点 key not in attrs
                    text3 = key
                    attrvalue =  kb.get(entity)[text3]
                    attrvalue = ','.join(attrvalue)
                    samples.append([text_id, text1, text3, attrvalue, 0])   # 同一实体的负样本
            
    print(f"length of sample: {len(samples)}")
    return samples


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
    os.makedirs(save_model_path,exist_ok=True)
    
    max_seq_len = args.max_seq_len
    gpu = args.gpu
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    nb_epochs = args.epochs

    device = torch.device(gpu)
    print("Loading dataset...")

    kb, entity_mapping = load_kb(args.kb_file)                 # head to relation to list(attr)
    train_data = load_data(args.train_file, kb)
    train_dataset = DatasetExtractor(train_data, max_seq_len, args.max_attrname_len, args.max_attrvalue_len,
                                     args.pretrain_model_path)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=False)   # 可以改为 shuffle=True
    
    dev_data = load_data(args.dev_file, kb)

    print('Creating model...')
    model = ExtractorModel(device=device, model_path=args.pretrain_model_path)
    print('Model created!')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # loss_fn = BCEFocalLoss(gamma=2, alpha=0.75, reduction='mean')
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
        
        all_ids, all_logits, all_labels= [],[],[]
        for i_batch, data in enumerate(loader):
            model.zero_grad()
            text_id, inputs, label = data
            token_ids = inputs["input_ids"].flatten(1).to(device)
            attention_mask = inputs["attention_mask"].flatten(1).to(
                device)
            token_type_ids = inputs["token_type_ids"].flatten(1).to(
                device)
            label = label.to(device)
            
            logit = model(
                token_ids=token_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(logit, label)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loader.set_postfix(
                Loss=running_loss / ((i_batch + 1) * batch_size),
                Epoch=epoch)
            loader.set_description('{}/{}'.format(epoch, nb_epochs))
            loader.update()
            
            all_ids.append(text_id.detach().cpu().numpy())
            all_logits.append(logit.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy())
        
        all_ids = np.concatenate(all_ids,)
        all_logits = np.concatenate(all_logits,)
        all_labels = np.concatenate(all_labels,)
        trn_auc, trn_acc =  cal_acc_score(all_ids, all_logits, all_labels)
        print("train step %d auc=%.6f, acc=%.6f"%(epoch+1, trn_auc, trn_acc))
        
        val_auc, val_acc  = validate(device, model, dev_data, args)
        print("valid step %d auc=%.6f, acc=%.6f"%(epoch+1, val_auc, val_acc))
        score = val_acc
        
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
    
    dataset = DatasetExtractor(dev_data, args.max_seq_len, args.max_attrname_len, args.max_attrvalue_len,
                                     args.pretrain_model_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    all_ids, all_logits, all_labels= [],[],[]
    
    for i_batch, data in enumerate(tqdm(dataloader)):
        model.zero_grad()
        text_id, inputs, label = data
        token_ids = inputs["input_ids"].flatten(1).to(device)
        attention_mask = inputs["attention_mask"].flatten(1).to(
            device)
        token_type_ids = inputs["token_type_ids"].flatten(1).to(
            device)
        label = label.to(device)

        logit = model(
            token_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
#         loss = loss_fn(logit, label)
        
        all_ids.append(text_id.detach().cpu().numpy())
        all_logits.append(logit.detach().cpu().numpy())
        all_labels.append(label.detach().cpu().numpy())
    all_ids = np.concatenate(all_ids,)
    all_logits = np.concatenate(all_logits,)
    all_labels = np.concatenate(all_labels,)

    auc, acc = cal_acc_score(all_ids, all_logits, all_labels)
    return (auc, acc)


# In[5]:


# os.environ["CUDA_VISIBLE_DEVICES"] = '0' # 0
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--train_file', type=str, default='../data/extractor_train.json')
parser.add_argument('--dev_file', type=str, default='../data/extractor_valid.json')
parser.add_argument('--kb_file', type=str, default='../../data/final_data/new_kg.json')
parser.add_argument('--pretrain_model_path', type=str, default='../../pretrain_model/roberta-retrained/')
parser.add_argument('--save_model_path', type=str, default='../model/intent_retrain_1')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_seq_len', type=int, default=64)
parser.add_argument('--max_attrname_len', type=int, default=20)
parser.add_argument('--max_attrvalue_len', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--epochs', type=int, default=5)  # 5
parser.add_argument('--validate_every', type=int, default=1)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--fp16', type=bool, default=False)
parser.add_argument("--fp16_opt_level", type=str, default="O1",
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html", )
parser.add_argument('--seed', type=int, default=1)

# sys.argv = ['']

args = parser.parse_args()

seed = args.seed
seed_everything(seed)

if args.mode == "train":
    train(args)
elif args.mode == "dev":
    pass
else:
    pass


# In[6]:


# Model created!
# 0/5: 100%|██████████| 6503/6503 [16:58<00:00,  6.39batches/s, Epoch=0, Loss=0.00415]
# train step 1 auc=0.930379, acc=0.800811
# 100%|██████████| 1525/1525 [01:13<00:00, 20.86it/s]
# valid step 1 auc=0.951145, acc=0.863569
# Validation accuracy 0.863569 increased from previous epoch, save best_model

# train step 1 auc=0.930570, acc=0.801412
# 100%|██████████| 1525/1525 [01:18<00:00, 19.42it/s]
# valid step 1 auc=0.950047, acc=0.860837
# Validation accuracy 0.860837 increased from previous epoch, save best_model


# In[7]:


# train step 2 auc=0.964714, acc=0.888817
# valid step 2 auc=0.954600, acc=0.874711

