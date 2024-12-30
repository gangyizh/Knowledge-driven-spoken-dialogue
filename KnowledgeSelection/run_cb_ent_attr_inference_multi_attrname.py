#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jieba
import jieba.posseg as pseg
import numpy as np
import json
import argparse
from collections import Counter
import copy
import re
import os
from tqdm import tqdm
import sys
import Levenshtein
import itertools
from rank_bm25 import BM25Okapi
from itertools import chain
from collections import Counter

from .NameEntityRecognition.ner_infere import NERInfere
# from IntentBinExtraction.extractor_infere import IntentInfere
from .EntitySelection.entselect_infere import EntInfere
# from AttrExtraction.extractor_infere import AttrInfere
from .IntentBinExtraction.extractor_infere import IntentInfere
from .utils import transform_attrname2inputname, transform_inputname2attrname
from .utils import load_kb, get_tail_kb, load_filter_kb,attrs2str,attrs2str_synonyms,is_entity_str,is_entity_str_addtion,kb_completion
from .utils import is_chinese_str,is_english_str,ch2pyinstr,get_pyin2ch
from .utils import CWDFinder,CHP_NAME,QuestionMatch,get_attrname2entities
from .utils import pre_process_query_snet,get_multi_query
from .post_konwledge import PostKnowledge

# from utils import inputname2synonyms


# In[2]:



def get_forbid_entity(kb):
    entities = list(kb.keys())
    attrnames = list(chain(*[list(kb[t].keys()) for t in kb.keys() ]))
    attrnames = set(attrnames)
    ea_entities = set(entities)&attrnames
    return ea_entities

def is_char_in_sent(w, sent):
    flag = True
    for c in w:
        if c not in sent:
            flag = False
            break
    return flag

def is_char_pyin_in_sent(w, sent):
    flag = True
    for c in w:
        if c not in sent:
            flag = False
            break
            
    if flag==False:
        sent_pyin_str = ch2pyinstr(sent)
        w_pyin_str = ch2pyinstr(w)
        if w_pyin_str in sent_pyin_str:
            flag = True
    return flag


def count_miss_char(w, sent):
    cnt = 0
    for c in w:
        if c in sent:
            cnt +=1
    n_miss = len(w) - cnt
    return n_miss

def is_cnt_char_in_sent(w, query, sent):
    cnt = 0
    for c in w:
        if c in sent:
            cnt +=1
    miss_threshold = (len(query)-1)//4 + 1
    n_miss = len(w) - cnt
    flag = True if n_miss<=miss_threshold else False
    return flag

def get_mapping_entiies(entities, kb, entity_mapping):
    all_entities = []

    for i, entity in enumerate(entities):
        if entity in kb.keys():
            all_entities.append(entity)

        if entity in entity_mapping:
            for x in entity_mapping.get(entity):
                if x in kb.keys():
                    all_entities.append(x)
    return  all_entities

def np_sigmoid(x):
    return 1/(1+np.exp(-x))


# In[3]:


kb, entity_mapping = load_filter_kb('data/final_data/new_kg.json')

filter_entities = get_forbid_entity(kb)

inputname2synonyms = {
    "开放什么游玩时间": ["什么时间","什么时候", "开放时间","几点"],
    "参观游玩多久时长": ["建议游玩时间","多久","预备"],
    "介绍":["特征特点","模样","时候"],   ### 只和生物有关
    "主要成就": ["有名气"],
    "诗词全文第一句最后一句背诵":["诗词全文","全文","第一句","最后一句","背诵"],
    "评分":["评价"],
    "出生日期":["岁数", "年纪"],
    "中心思想":["情感"],
    "作品赏析":["点评","评价"],
    "赏析":["点评","评价"],
    "朝代":["时候"],
    "作者简介":["生平","字名号"],
    "外文名":["原始名"],
    "是否具备观赏价值":["好看"]
    # "简介":["生平"]
}
# special_enties = ['0', '1', '1040', '119', '13', '15', '16', '1701', '1906', '1941', '1952', '1965', '1999', '20', 
#                   '2001年9月11日', '2002', '2006', '2008', '2012', '2046', '21', '224', '23', '25', '3', '300', '31', '35',
#                   '42', '5', '52', '8', '80', '9', '90', 'O', 'W', 'w', '一', '兰', '句', '叶', '她', '家', '弟', '扇', '春', 
#                   '杏', '枣', '柽', '梨', '榆', '榴', '槿', '满', '燕', '爱', '画', '竹', '羊', '芦', '花', '葵', '蚊', '蝉', 
#                   '赢', '雨', '雪', '韭', '飘', '鸟', '？']
special_enties = [ '一', '兰', '句', '叶', '她', '家', '弟', '扇', '春', '杏', '枣', '柽', '梨', '榆', '榴', '槿', '满', '燕', 
                  '爱', '画', '竹', '羊', '芦', '花', '葵', '蚊', '蝉', '赢', '雨', '雪', '韭', '飘', '鸟'] # '？'
special_enties = set(special_enties)


# In[4]:


class KnowledgeSelection():
    def __init__(self, args):
        self.args = args
        self.kb, self.entity_mapping = load_filter_kb(args.kb_file)
        # self.extra_kb, self.extra_entity_mapping = load_kb(args.extra_kb_file)
        # self.kb = kb_completion(self.kb, self.extra_kb)
        self.tail_kb = get_tail_kb(self.kb, self.entity_mapping)
        self.all_kb = {**self.kb, **self.tail_kb}
        self.attrname2entities = get_attrname2entities(self.kb)
        self.pyin2ch =get_pyin2ch(self.kb, self.entity_mapping, self.tail_kb)
        self.chp_jude = CHP_NAME()
        self.qa_macth = QuestionMatch(inputfile='data/final_data/Noise-free/train.json')
        
        self.poetry_kb,_ = load_kb('data/Knowledge_Graph_Data/Ancient Chinese Poetry/Modified Data/shici_final.json')
        
        self.post_processer = PostKnowledge()
        
        self.ner_infere = NERInfere(args.gpu, args.tag_file,
                                    args.ner_pretrain_model_path,
                                    args.ner_save_model_path,
                                    args.ner_max_seq_len)
        
        self.ent_infere = EntInfere(args.gpu,
                                      args.ent_select_pretrain_model_path,
                                      args.ent_select_save_model_path,
                                      args.max_conv_seq_len,
                                      args.max_entity_len,
                                      args.max_attrname_len,
                                      args.max_attrvalue_len,
                                         )
        
        attr_pretrain_model_path = [
            # "../pretrain_model/roberta-retrained/",
            "pretrain_model/chinese-macbert-base/",
            # "../pretrain_model/chinese-roberta-wwm-ext/",
            "pretrain_model/ernie-1.0-base-zh/",
            # "../pretrain_model/chinese-pert-base/",
        ]
        attr_save_model_path = [
            # "model/intent_retrain_1/",
            "KnowledgeSelection/model/intent_macbert_1/",
            # "model/intent_roberta_wwm_1/",
            "KnowledgeSelection/model/intent_erniezh_1/",
            # "model/intent_pert_1/",
        ]
        
        self.attr_infere_models = []
        for i in range(len(attr_save_model_path)):
            attr_infere = IntentInfere(args.gpu,
                                          attr_pretrain_model_path[i],
                                          attr_save_model_path[i],
                                          args.max_seq_len,
                                          # args.max_entity_len,
                                          args.max_attrname_len,
                                          args.max_attrvalue_len,
                                         )
            self.attr_infere_models.append(attr_infere)
        
        
        self.kb_enitites = list(self.kb.keys())
        self.tail_kb_enitites = list(self.tail_kb.keys())
        
        self.all_kb_entity = list(self.kb.keys()|self.entity_mapping.keys()|self.tail_kb.keys())
        self.idf = {}
        
        self.tokenized_corpus = [w for w in self.all_kb_entity]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.cwdfinder =  CWDFinder(self.kb,)
        for word in list(self.kb.keys()|self.entity_mapping.keys()):
            jieba.add_word(word, 100, "entity")

    def get_idf(self, sentences):
        idf = Counter()
        for sent in sentences:
            words = jieba.lcut(sent)
            words = list(set(words))
            idf.update(words)
        for key in idf:
            idf[key] = len(sentences) / idf[key]
        return idf

    def load_valid_data(self, valid_file):
        with open(valid_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        samples = []
        all_messages = []
        for sample in data:
            messages = sample.get("messages")
            previous_message = messages[0].get("message")
            all_messages.append(previous_message)
            context = [previous_message]
            prev_entities = []
            for i in range(1, len(messages)):
                message = messages[i].get("message")
                all_messages.append(message)
                if "attrs" in messages[i]:
                    attrs = messages[i].get("attrs")
                    qsample = dict(question=previous_message, answer=message,
                                   knowledge=attrs, context=copy.deepcopy(context),
                                   prev_entities=list(set(prev_entities)))
                    if previous_message.endswith("？"):
                        samples.append(qsample)
                    prev_entities.extend([attr.get("name") for attr in attrs])
                context.append(message)
                previous_message = message
        # self.idf = self.get_idf(all_messages)
        return samples

    def load_test_data(self, test_file):
        with open(test_file, 'r', encoding='utf-8') as fin:
            data = json.load(fin)

        samples = {}
        all_messages = []
        for index in data:
            if len(data[index])==0:
                break
            question = data[index][-1].get("message")
            context = [turn["message"] for turn in data[index]]
            all_messages.extend(context)
            sample = {"question": question, "context": context}
            samples[index] = sample
        # self.idf = self.get_idf(all_messages)
        return samples

    def get_entity_by_jieba(self, context):
        candidates = []
        for seq in context:
            seq = seq.replace("。","")
            words = pseg.cut(seq)
            # print(words)
            for (word, pos) in words:
                if pos == "entity":
                    candidates.append(word)

        pred_words = {}
        for word in candidates:
            if word not in self.all_kb_entity:
                continue
            s = self.idf.get(word, 5)
            pred_words[word] = s

        pred_words = dict(
            sorted(pred_words.items(), key=lambda x: x[1], reverse=True))
        return list(pred_words.keys())[:1]
    
    def get_multi_entity_by_jieba(self, context):
        candidates = []
        for seq in context:
            seq = seq.replace("。","")
            if len(seq)>=40:
                continue
            words = pseg.cut(seq)
            # print(words)
            for (word, pos) in words:
                if pos == "entity" or word in special_enties:
                    candidates.append(word)

        candidates = [x for x in candidates if (is_entity_str(x) and len(x)>=2) or x in special_enties]
        candidates = list(set(candidates))
        
        return candidates
    
    
    def get_bm25_match(self, query, sent, context, cidx, threshold=5):
        scores = self.bm25.get_scores(query)
        best_docs = sorted(list(zip(self.tokenized_corpus, scores)), key=lambda x: x[1], reverse=True)[:10]
        
        
        match = []
        ### 相同名字
        if is_chinese_str(query):
            pyin_str = ch2pyinstr(query)
            pyin_macth_words = self.pyin2ch.get(pyin_str, [])
            match.extend(pyin_macth_words)
            if self.args.debug and len(pyin_macth_words)>0:
                print("pyin match:", pyin_macth_words)
            if len(pyin_macth_words)==1 and len(pyin_macth_words[0])>=4:  ### 严格语音纠正
                context[cidx] = context[cidx].replace(query, pyin_macth_words[0])
                if self.args.debug:
                    print(f"pyin correct: {sent} => {context[cidx]}")
                    
        ### 多字的情况
        for (w,s) in best_docs:
            if w in sent and len(w)<len(query):   ### todo: 考虑候选的匹配度
                match.append(w)
                break
        
        ### 缺字的情况,若多个满足筛选条件，则不考虑
        miss_words = []
        for (w,s) in best_docs:
            if is_cnt_char_in_sent(w, query, sent) and len(w)>len(query) and is_entity_str(w):  
                miss_words.append(w)
        if self.args.debug:
            print("miss_words:", miss_words)
        if len(miss_words)>=1:
            miss_word_count = [(w, count_miss_char(w,sent)) for w in miss_words]
            miss_word_count = sorted(miss_word_count, key=lambda x: x[1], )
            match.append(miss_word_count[0][0])
        
        ### 其他情况
        if best_docs[0][1]>best_docs[1][1]+ threshold: # 
            match.append(best_docs[0][0])
            
        match = list(set(match))
        
        ### 包含关系过滤
        match = sorted(match, key=lambda x: len(x), reverse=True)
        filter_macth = []
        
        for i in range(len(match)):
            flag = True
            for j in range(0,i):
                if (match[j] in sent or re.sub('[·。 ]', "", match[j]) in sent) and match[i] in match[j]:
                    flag = False
                    break
            if flag:
                filter_macth.append(match[i])
        
        if len(filter_macth)>0:
            return filter_macth
        else:
            return None

    def get_last_bm25_match(self, query, sent, context, cidx, threshold=0):
        scores = self.bm25.get_scores(query)
        best_docs = sorted(list(zip(self.tokenized_corpus, scores)), key=lambda x: x[1], reverse=True)[:10]
        match = [x[0] for x in best_docs[:5]]
        match = list(set(match))
        if len(match)>0:
            return match
        else:
            return None
    
    def get_bm25_better_word(self, word, sent, n_cand=15):
        scores = self.bm25.get_scores(word)
        best_words = sorted(list(zip(self.tokenized_corpus, scores)), key=lambda x: x[1], reverse=True)[1:n_cand+1]
        best_words = [t[0] for t in best_words]
        for w in best_words:
            # if is_char_in_sent(w, sent) and len(w)>len(word):
            # if is_cnt_char_in_sent(w, word, sent) and len(w)>len(word):
            if is_char_pyin_in_sent(w, sent) and len(w)>len(word):
                return w        
        return None
    
    
    def get_entities_bm25_tag(self, context):
        entities = []           # 整段对话的实体
        all_pred_entities = []  
        question_entities = []  # 最后文本的实体   暂时未用到
        
        ### 遍历整段对话，识别实体
        for cidx,sent in enumerate(context):
#             if cidx%2==1 and cidx>1:
#                 continue
            ### todo: 写好符号列表
            sent = sent.replace('。','')
            pred_entities = self.ner_infere.ner(sent)
            if self.args.debug:
                print(sent, pred_entities)
            if len(pred_entities)==1 and is_english_str(pred_entities[0]):
                pred_entities = re.findall('[a-zA-Z]+', sent)
            
            for pred in pred_entities:
                if pred == "" or pred in self.attrname2entities:
                    continue
                pred = pred.replace('。', '')
                if pred in self.all_kb_entity:
                    better_word = self.get_bm25_better_word(pred, sent)
                    if better_word is not None:
                        if pred in better_word:
                            pred = better_word
                        else:
                            entities.append(better_word)
                        
                    if pred in filter_entities:
                        continue
                        
                    entities.append(pred)
                    
#                     if cidx == len(context) - 1:
#                         question_entities.append(pred)
                    # print(pred)
                elif len(pred)>=2:
                    macth_pred = self.get_bm25_match(pred, sent, context, cidx, threshold=self.args.bm25_threshold)
                    if macth_pred is not None:
                        entities.extend(macth_pred)
#                         if cidx ==len(context) - 1:
#                             question_entities.extend(macth_pred)
                # all_pred_entities.append(pred)
        
        entities = [t for t in entities if is_entity_str(t) or t in special_enties]
        # question_entities = [t for t in question_entities if is_entity_str(t) or t in special_enties]
        
        entities2count = {} #Counter(entities)
        # all_pred_entities = list(set(all_pred_entities))     # 加上实体是 数字字符串 直接删除
        
        
        r_entities = []
        for t in range(len(entities)-1, -1, -1):
            if entities[t] not in r_entities:
                r_entities.append(entities[t])
        # question_entities = list(set(question_entities))
        
        ### 最后问题的实体被提起两次则直接该实体，
        ### 加进去效果很差，一些短语干扰，所以将实体长度改成大于等于4，提高筛选
#         if len(question_entities)!=0:
#             hard_entities = []
#             for t in question_entities:
#                 if entities2count.get(t)>=2 and len(t)>=3:  
#                     hard_entities.append(t)
#             if len(hard_entities)>0:
#                 return hard_entities, all_pred_entities, "last_hard_ner"


#         if len(entities) == 0:
#             r_entities = []
#             jieba_entities = self.get_entity_by_jieba(context)
#             r_entities.extend(jieba_entities)
            
#             return r_entities, all_pred_entities, entities2count, question_entities,"jieba&&bm25"
#         else:
#             return r_entities, all_pred_entities, entities2count, question_entities, "ner"
        
        jieba_entities = self.get_multi_entity_by_jieba(context)
        if self.args.debug:
            print("jieba_entities:", jieba_entities)
        if len(jieba_entities)>0:
            for x in jieba_entities:
#                 if x not in r_entities:
#                     r_entities.append(x)
                flag = True
                for ent in r_entities:
                    if x in ent:
                        flag = False
                        break
                if flag:
                    r_entities.append(x)
        
        return r_entities, all_pred_entities, entities2count, question_entities, "ner&&jibea"

        
    def get_last_entities_bm25_tag(self, context):
        entities = []           # 整段对话的实体
        # all_pred_entities = []  
        question_entities = []  # 最后文本的实体   暂时未用到
        
        ### 遍历整段对话，识别实体
        for cidx,sent in enumerate(context):
            sent = sent.replace('。','')
            pred_entities = self.ner_infere.ner(sent)
            if self.args.debug:
                print(sent, pred_entities)
            
            for pred in pred_entities:
                if pred == "" or pred in self.attrname2entities:
                    continue
                pred = pred.replace('。', '')
                if len(pred)>=2:
                    macth_pred = self.get_last_bm25_match(pred, sent, context, cidx)
                    if macth_pred is not None:
                        entities.extend(macth_pred)
#                         if cidx ==len(context) - 1:
#                             question_entities.extend(macth_pred)
                # all_pred_entities.append(pred)
        
        entities = [t for t in entities if is_entity_str(t) or t in special_enties]
        # question_entities = [t for t in question_entities if is_entity_str(t) or t in special_enties]
        entities2count = {} # Counter(entities)
        all_pred_entities = list(set(all_pred_entities))     # 加上实体是 数字字符串 直接删除
        r_entities = []
        for t in range(len(entities)-1, -1, -1):
            if entities[t] not in r_entities:
                r_entities.append(entities[t])
        # question_entities = list(set(question_entities))

        return r_entities, all_pred_entities, entities2count, question_entities,"last_bm25"
    
#     def entities_attr_filter(self, all_entities, question):    ### 这个过滤方法，会过滤掉正确答案，应该更严格设计
#         ### 按实体属性有无在问题中过滤，如果过滤后列表为空，则返回原列表
#         filter_entities = []
#         for entity in all_entities:
#             flag = False
#             for attr in self.all_kb[entity]:
#                 if attr:
#                     flag = True
#                     break
#             if flag:
#                 filter_entities.append(entity)
                
#         if len(filter_entities)>0:                
#             return filter_entities
#         else:
#             return all_entities
        
    
    def get_ent_attr_intent(self, entities, query, context):
        if len(entities)==0:
            return None, None
        query = query.replace("。","")
        query = pre_process_query_snet(query)
        all_attrnames = []
        for ent in entities:
            # attrnames,attrvalues = attrs2str(self.all_kb, ent)
            attrnames,attrvalues = attrs2str_synonyms(self.all_kb, ent, inputname2synonyms)  # 改变
#             if ent in self.poetry_kb:
#                 attrnames = "这首诗" + attrnames
            all_attrnames.append(attrnames)
            
        
        pred_ent_score = self.ent_infere.text_smiliary([context]*len(entities), entities, all_attrnames, [""]*len(entities),bacth_size = 4)
        pred_ent_score = pred_ent_score.squeeze(-1)
        ent_index = np.argmax(pred_ent_score)
        pred_entities = entities[ent_index]
        select_entities = [pred_entities]
        
        ### 强插机制
        for ent in entities:
            if ent in query and ent not in select_entities and is_chinese_str(ent) and ent not in pred_entities:
                if len(ent)>=3 or self.chp_jude.is_people_name(ent): 
                    select_entities.append(ent)
        if len(select_entities)>1: 
            select_entities = select_entities[::-1]
        
        ### ()问题
        if select_entities[0] in self.entity_mapping:
            for x in self.entity_mapping[select_entities[0]]:
                select_entities.append(x)
#        如果实体属性在问题中出现，将实体加入候选
#         attr_in_query_entities = []
#         for ent in entities:
#             flag = False
#             for attr in self.all_kb[ent]:
#                 if attr in query:
#                     flag = True
#             if flag:
#                 attr_in_query_entities.append(ent)
#         if pred_entities not in attr_in_query_entities:
#             select_entities.extend(attr_in_query_entities)
            
#         尝试筛选2个实体进入排序            
#         topk = 2
#         ent_index = np.argsort(pred_ent_score.squeeze(-1))[::-1][:topk]
#         select_entities = [entities[int(t)] for t in ent_index]
        
        if self.args.debug:
            print("entity score:", list(zip(entities, pred_ent_score)))
            print("select_entities:", select_entities)
        
        candidates = []                  # 所有实体的所有attrnames
        input_candidates = []
        candidates_attrvalues = []
        global_entities = []
        for entity in select_entities:
#             attrs = list(self.kb.get(entity, {}).keys())
#             candidates.extend(attrs)
            entity_attrs = self.all_kb.get(entity, {})
            attr_count = 0
            for attr,attrvalue in entity_attrs.items():
                candidates.append(attr)
                input_candidates.append(attr)
                attrvalue_str = ','.join([str(t) for t in attrvalue])
                candidates_attrvalues.append(attrvalue_str)    # 这里置空了，导致效果不好
                attr_count +=1
                
                if attr in inputname2synonyms:
                    for x in inputname2synonyms[attr]:
                        candidates.append(attr)
                        input_candidates.append(x)
                        candidates_attrvalues.append(attrvalue_str)
                        attr_count +=1
                        
            global_entities.extend([entity] * attr_count)
        if len(candidates) == 0:
            return None, None
        
        if len(select_entities)==1:
            query = query.replace(select_entities[0],"ne")      ### 在推断时同样遮盖实体
        # pred_attr_score = self.attr_infere.text_smiliary([query]*len(candidates), input_candidates, candidates_attrvalues, bacth_size = 16)
        
        pred_attr_score = np.zeros([len(candidates),1])
        for attr_model in self.attr_infere_models:
            attr_score = attr_model.text_smiliary([query]*len(candidates), input_candidates, candidates_attrvalues, bacth_size = 16)
            pred_attr_score += np_sigmoid(attr_score)/len(self.attr_infere_models)
        pred_attr_score = pred_attr_score.squeeze(1)
        if self.args.debug:
            print(list(zip(global_entities,candidates,pred_attr_score,candidates_attrvalues)))
        
        attrname_threshold = 0.9
        if len(pred_attr_score)==1:
            attr_index = np.argmax(pred_attr_score)
            pred_intent = candidates[attr_index]
            pred_entities = global_entities[attr_index]
            return [pred_intent], pred_entities
        else:
            top1_idx, top2_idx = np.argsort(pred_attr_score)[::-1][:2]
            # print(type(top1_idx))
            # print(top1_idx)
            # print(top2_idx)
            if candidates[top1_idx]!=candidates[top2_idx] and pred_attr_score[top1_idx]>0.9 and \
                pred_attr_score[top2_idx]>0.9 and global_entities[top1_idx]==global_entities[top2_idx]:

                pred_intent_1 = candidates[top1_idx]
                pred_intent_2 = candidates[top2_idx]
                pred_entities = global_entities[top1_idx]
                return [pred_intent_1, pred_intent_2], pred_entities
            else:
                attr_index = top1_idx
                pred_intent = candidates[attr_index]
                pred_entities = global_entities[attr_index]
                return [pred_intent], pred_entities


    def get_pred_knowledge(self, entity, intent):
        if entity is None:
            return []
        pred_knowledge = []
        if entity not in self.all_kb:
            return []
        if intent not in self.all_kb.get(entity):
            print(f"{intent} not in {self.kb.get(entity)}")
            return []
        
        for value in self.all_kb.get(entity)[intent]:
#             if intent == "简介":
#                 intent = "Information"
            intent = transform_inputname2attrname(intent)
            if entity in self.kb.keys():
                known = {"name": entity, "attrname": intent, "attrvalue": value}
            else:
                known = {"name": value, "attrname": intent, "attrvalue": entity, "know_reverse":1}
                
            pred_knowledge.append(known)
        return pred_knowledge

    def _match(self, gold_knowledge, pred_knowledge):
        result = []
        for pred in pred_knowledge:
            matched = False
            for gold in gold_knowledge:
                if isinstance(pred["attrvalue"], list):
                    pred_attrvalue = " ".join(sorted(pred["attrvalue"]))
                else:
                    pred_attrvalue = pred["attrvalue"]
                if isinstance(gold["attrvalue"], list):
                    gold_attrvalue = " ".join(sorted(gold["attrvalue"]))
                else:
                    gold_attrvalue = gold["attrvalue"]
                if pred['name'] == gold['name'] and pred['attrname'] == gold[
                    'attrname'] and pred_attrvalue == gold_attrvalue:
                    matched = True
            result.append(matched)
        return result

    def calu_knowledge_selection(self, gold_knowledge, pred_knowledge):
        if len(gold_knowledge) == 0 and len(pred_knowledge) == 0:
            return 1.0, 1.0, 1.0

        precision, recall, f1 = 0.0, 0.0, 0.0
        relevance = self._match(gold_knowledge, pred_knowledge)
        if len(relevance) == 0 or sum(relevance) == 0:
            return precision, recall, f1

        tp = sum(relevance)
        precision = tp / len(pred_knowledge) if len(
            pred_knowledge) > 0 else 0.0
        recall = tp / len(gold_knowledge) if len(gold_knowledge) > 0 else 0.0
        if precision == 0 and recall == 0:
            return precision, recall, f1
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def evaluate(self, datafile, outputfile):
        data = self.load_valid_data(datafile)

        total = len(data)
        metrics = {"p": 0, "r": 0, "f1": 0}
        eval_samples = []
        for sample in tqdm(data):
            knowledge = sample.get("knowledge")
            context = sample.get("context")
            question = sample.get("question")
            answer = sample.get("answer")
            
            cwd_res = self.cwdfinder.run(question)
            ### 成语规则
            if cwd_res is not None and len(cwd_res)>0:
                pred_knowledges = []
                for entity, intent in cwd_res:
                    pred_knowledge = self.get_pred_knowledge(entity, intent)
                    pred_knowledges.extend(pred_knowledge)
                if len(pred_knowledges)>0:
                    p, r, f1 = self.calu_knowledge_selection(knowledge, pred_knowledges)
                    eval_sample = {"question": question, "context": context,
                              "groud_attrs": knowledge, "pred_attrs":pred_knowledges,"groud_response":answer,
                              "entities":[], "tag":"cwd", "pred_entities":[]}
                    eval_samples.append(eval_sample)
                    metrics["p"] += p
                    metrics["r"] += r
                    metrics["f1"] += f1
                    continue
            
            
            entities, pred_entities, entities2count, question_entities, tag = self.get_entities_bm25_tag(context)
            if len(entities)==0:
                entities, pred_entities, entities2count, question_entities, tag = self.get_last_entities_bm25_tag(context)
                
            all_entities = get_mapping_entiies(entities, self.all_kb, self.entity_mapping)
            # all_question_entities = get_mapping_entiies(entities, self.all_kb, self.entity_mapping)
            
            intent, entity = self.get_ent_attr_intent(all_entities, question, context)
            pred_knowledge = self.get_pred_knowledge(entity, intent)        # 根据 entity 和 intent查找值
            p, r, f1 = self.calu_knowledge_selection(knowledge, pred_knowledge)
            eval_sample = {"question": question, "context": context,
                      "groud_attrs": knowledge, "pred_attrs":pred_knowledge,"groud_response":answer,
                      "entities":entities, "tag":tag, "pred_entities":pred_entities}
            eval_samples.append(eval_sample)
            
            metrics["p"] += p
            metrics["r"] += r
            metrics["f1"] += f1

        for key in metrics:
            metrics[key] = metrics.get(key) / total
        print(metrics)
        
        with open(outputfile, 'w', encoding='utf-8') as fout:
            json.dump(eval_samples, fout, ensure_ascii=False)
        
        return metrics
    
    
    def inference_konwledge(self, context):
        question = context[-1]

        ### 历史问句匹配
        match_attrs = self.qa_macth.run(question)
        if len(context)==1 and match_attrs is not None:
            sample = {"question": question, "context": context,
                  "attrs": match_attrs}
            if self.args.debug:
                print("question match!!!")
                print("question:", question)
                print("pred_knowledges:", pred_knowledges)
            return sample

        ### 成语规则
        cwd_res = self.cwdfinder.run(question)
        if cwd_res is not None and len(cwd_res)>0:
            pred_knowledges = []
            for entity, intent in cwd_res:
                pred_knowledge = self.get_pred_knowledge(entity, intent)
                pred_knowledges.extend(pred_knowledge)
            for pred_knowledge in pred_knowledges:
                pred_knowledge["know_reverse"] = 1
            sample = {"question": question, "context": context,
                      "attrs": pred_knowledges}
            if self.args.debug:
                print("question:", question)
                print("pred_knowledges:", pred_knowledges)
            return sample


        entities, pred_entities, entities2count, question_entities, tag = self.get_entities_bm25_tag(context)
        if len(entities)==0:
            if self.args.debug:
                print("starting last_entities_bm25 match...")
            entities, pred_entities, entities2count, question_entities, tag = self.get_last_entities_bm25_tag(context)
        if self.args.debug:
            print("get_entities_bm25_tag: ", entities)

#             if len(entities)==0:
#                 if args.debug:
#                     print("starting jieba match...")
#                 entities, pred_entities, entities2count, question_entities, tag = self.get_last_entities_bm25_tag(context)
#                 if args.debug:
#                     print("jieba: ", entities)   

        all_entities = get_mapping_entiies(entities, self.all_kb, self.entity_mapping)
        # all_question_entities = get_mapping_entiies(entities, self.all_kb, self.entity_mapping)
        if self.args.debug:
            print("entity mapping: ", all_entities)

        # all_entities = self.entities_attr_filter(all_entities, question)
        # if args.debug:
        #     print("entity attr filter: ", all_entities)

        multi_query = get_multi_query(question)
        # print(multi_query)
        if len(multi_query)<2:
            intents, entity = self.get_ent_attr_intent(all_entities, question, context)  # intents list
            pred_knowledge = []
            for intent in intents:
                pred_know = self.get_pred_knowledge(entity, intent)
                pred_knowledge.extend(pred_know)
        else:
            pred_knowledge = []
            pred_intent_entity = []
            for query in multi_query:
                intents, entity = self.get_ent_attr_intent(all_entities, query, context) # intents list
                for intent in intents:
                    intent_entity_str = intent+"_"+entity
                    # print(intent, entity)
                    if intent_entity_str not in pred_intent_entity:
                        pred_intent_entity.append(intent_entity_str)
                        pred_know = self.get_pred_knowledge(entity, intent)
                        pred_knowledge.extend(pred_know)

        if self.args.debug:
            print(entity, intent)
            print(pred_knowledge)
        sample = {"question": question, "context": context,
                  "attrs": pred_knowledge, "entities":entities,
                 "tag":tag, "pred_entities":pred_entities}
        sample = self.post_processer.run(sample)
        return sample
        
    
    
    def test(self, datafile, outputfile, test_index=None):
        data = self.load_test_data(datafile)

        samples = {}
        for index in tqdm(data):
            if  self.args.debug and test_index is not None and index!=str(test_index):
                continue
            question = data.get(index).get("question")
            context = data.get(index).get("context")
            
            sample = self.inference_konwledge(context)
            samples[index] = sample

        with open(outputfile, 'w', encoding='utf-8') as fout:
            json.dump(samples, fout, ensure_ascii=False)



# In[5]:



# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str,
#                     default="test")

# parser.add_argument('--tag_file', type=str,
#                     default="data/tag.txt")
# parser.add_argument('--ner_pretrain_model_path', type=str,
#                     default="../pretrain_model/chinese-roberta-wwm-ext")
# parser.add_argument('--ner_save_model_path', type=str,
#                     default="model/ner/")
# parser.add_argument('--gpu', type=int, default=0)
# parser.add_argument('--ner_max_seq_len', type=int, default=512)

# # parser.add_argument('--extractor_pretrain_model_path', type=str,
# #                     default="pretrain_model/ernie-1.0/")
# # parser.add_argument('--extractor_save_model_path', type=str,
# #                     default="model/intent")

# parser.add_argument('--ent_select_pretrain_model_path', type=str,
#                     default="../pretrain_model/ernie-3.0-base-zh/")
# parser.add_argument('--ent_select_save_model_path', type=str,
#                     default="model/entity_select_ernie3/")

# parser.add_argument('--attr_extract_pretrain_model_path', type=str,
#                     default="../pretrain_model/")
# parser.add_argument('--attr_extract_save_model_path', type=str,
#                     default="model/intent_retrain_1/")

# parser.add_argument('--extractor_max_seq_len', type=int, default=50)
# parser.add_argument('--max_conv_seq_len', type=int, default=400)
# parser.add_argument('--max_seq_len', type=int, default=64)
# parser.add_argument('--max_entity_len', type=int, default=40)  # 多实体
# parser.add_argument('--max_attrname_len', type=int, default=40)
# parser.add_argument('--max_attrvalue_len', type=int, default=40)

# parser.add_argument('--bm25_threshold', type=float, default=2)
# parser.add_argument('--kb_file', type=str,
#                     default="../data/final_data/new_kg.json")
# parser.add_argument('--extra_kb_file', type=str,
#                     default="../data/Comparison_Data/Knowledge_Graph_Data/Marine_Fishes/Modified_Data/fish_final_v2_16.json")
# parser.add_argument('--valid_file', type=str,
#                     default="../data/final_data/Noise-added/valid.json")
# parser.add_argument('--test_file', type=str,
#                     default="../data/final_data/test.json")
# parser.add_argument('--result_file', type=str,
#                     default="test_know_result.json")
# parser.add_argument('--val_result_file', type=str,
#                     default="eval_know_result.json")

# parser.add_argument('--debug', type=bool,
#                     default=False)

# sys.argv = ['']
# t_args = parser.parse_args()

# selector = KnowledgeSelection(t_args)
# if t_args.mode == "test":
#     selector.test(t_args.test_file, t_args.result_file, test_index=102)
# else:
#     selector.evaluate(t_args.valid_file, t_args.val_result_file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


### {'p': 0.7276130019981182, 'r': 0.7598732603672693, 'f1': 0.7277000286985552} pert
### {'p': 0.7332888481196234, 'r': 0.7659445137582621, 'f1': 0.7337336197542192} chinese-roberta-wwm-ext
### {'p': 0.7360266680741765, 'r': 0.767914042372735, 'f1': 0.7357241571872623} macbert
### {'p': 0.7352520368707036, 'r': 0.7691438090323943, 'f1': 0.7359585652942794} erniezh1


# In[7]:


### {'p': 0.733349911014405, 'r': 0.7665970040050345, 'f1': 0.7337462866472727}
### {'p': 0.7325791170966698, 'r': 0.7660013905231482, 'f1': 0.7330689222953236}


# In[ ]:





# In[ ]:




