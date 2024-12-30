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

    
def load_data(datafile, kb, entity_mapping, bm25_model, neg_cnt, ner_infere=None):
    
    data = []
    with open(datafile, 'r', encoding='utf-8') as fin:
        for line in fin:
            data.append(json.loads(line.strip()))
    print(f"length of data: {len(data)}")
    
    kb_enitites = list(kb.keys()|entity_mapping.keys())
    attrname2entities = get_attrname2entities(kb)
    
    samples = []
    for text_id,sample in enumerate(tqdm(data)):
        # if text_id>1000:
        #     break
        query = sample.get("question")
        context = sample.get("context")
        # {question: str，answer: str，knowledge: list(dict), context: list(str)，prev_entities: list(str)}
        entity2attr = {}     # answer中使用到的 entity to set(attrname)
        for known in sample.get("knowledge"):
            entity = known.get("name")
            attrname = known.get("attrname")
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
            if len(subgraph) == 0:
                continue
            # text1 = query.replace(entity, "ne")   # 不理解替换的意思
            text1 = context
            for attr in attrs:
                text2 = attr
                attrnames,attrvalues = attrs2str(kb, entity)
                samples.append([text_id, text1, entity,attrnames,attrvalues, 1, 1])       # (id, context, attrname, attrvalue)正样本
                ### 正确attrname，错误entity的负样本
                neg_entities = []
                bcnt = 0
                while (len(neg_entities)==0 or entity in neg_entities) and len(attrname2entities[attr])>neg_cnt:
                    neg_entities = random.sample(attrname2entities[attr], neg_cnt)
                    bcnt +=1
                    if bcnt>3:
                        neg_entities = []
                        break
                
                for neg_ent in neg_entities:
                    attrnames,attrvalues = attrs2str(kb, neg_ent)
                    samples.append([text_id, text1, neg_ent, attrnames,attrvalues, 0, 1])
                    

#             for key in subgraph:
#                 if key not in attrs:    # 优化点 key not in attrs
#                     text3 = key
#                     attrvalue =  kb.get(entity)[text3]
#                     attrvalue = ','.join(attrvalue)
#                     samples.append([text_id, text1, entity, text3, attrvalue, 1, 0])   # 同一实体的负样本
         
        if ner_infere is not None:
            all_pred_entities = []
            for sent in context:
                pred_entities = ner_infere.ner(sent)
                for pred in pred_entities:
                    if pred == "":
                        continue
                    if pred in kb_enitites:
                        all_pred_entities.append(pred)
                    elif len(pred)>=2:
                        macth = bm25_model.get_bm25_match(pred, sent, threshold=1.0)   ### 调小阈值加入更多对话中的负样本
                        if macth is not None and len(macth)>0:
                            all_pred_entities.extend(macth)
#                             print("add bm25 entity")
            all_pred_entities = sorted(list(set(all_pred_entities)))
            
            all_entities = []
            for i, entity in enumerate(all_pred_entities):
                if entity in kb.keys():
                    all_entities.append(entity)
                    
                if entity in entity_mapping:
                    for x in entity_mapping.get(entity):
                        all_entities.append(x)
            
            for ent in all_entities:
                if ent not in entity2attr:
#                     for attrname, attrvalue in kb[ent].items():
#                         if attrname == "Information":
#                             attrname = "简介"
#                         attrvalue = ','.join(attrvalue)
                    attrnames,attrvalues = attrs2str(kb, ent)
                    samples.append([text_id, text1, ent, attrnames,attrvalues, 0, 0])   # 不同实体的负样本
                    # print("add negetive sample")
    
    print(f"length of sample: {len(samples)}")
    return samples


def prepare_data(args):
    gpu = args.gpu
    
    device = torch.device(gpu)
    print("Loading dataset...")

    kb, entity_mapping = load_kb(args.kb_file)                 # head to relation to list(attr)
    bm25_model = BM25_Macth(kb, entity_mapping)
    ner_infere = NERInfere(args.gpu, args.tag_file,
                        args.ner_pretrain_model_path,
                        args.ner_save_model_path,
                        args.ner_max_seq_len)
    train_data = load_data(args.train_file, kb, entity_mapping, bm25_model, args.neg_cnt, ner_infere)
    
    dev_data = load_data(args.dev_file, kb, entity_mapping, bm25_model, args.neg_cnt, ner_infere)
    with open(args.train_data_file,'w') as f:
        json.dump(train_data, f,ensure_ascii=False)
    with open(args.val_data_file,'w') as f:
        json.dump(dev_data, f, ensure_ascii=False)




if __name__=="__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag_file', type=str,
                        default="../data/tag.txt")
    parser.add_argument('--ner_pretrain_model_path', type=str,
                        default="../../pretrain_model/chinese-roberta-wwm-ext")
    parser.add_argument('--ner_save_model_path', type=str,
                        default="../model/ner")
    parser.add_argument('--ner_max_seq_len', type=int, default=512)


    parser.add_argument('--train_file', type=str, default='data/extractor_train.json')
    parser.add_argument('--dev_file', type=str, default='data/extractor_valid.json')
    parser.add_argument('--kb_file', type=str, default='../../data/final_data/new_kg.json')

    parser.add_argument('--train_data_file', type=str, default='data/train_data.json')
    parser.add_argument('--val_data_file', type=str, default='data/val_data.json')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_conv_seq_len', type=int, default=400)
    # parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--max_entity_len', type=int, default=10)
    parser.add_argument('--max_attrname_len', type=int, default=40) # 多实体
    parser.add_argument('--max_attrvalue_len', type=int, default=40)
    parser.add_argument('--neg_cnt', type=int, default=4)

    args = parser.parse_args()

    seed = 1
    seed_everything(seed)
    prepare_data(args)

