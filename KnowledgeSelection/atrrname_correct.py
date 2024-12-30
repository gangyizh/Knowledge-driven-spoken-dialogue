#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pypinyin import lazy_pinyin
from copy import deepcopy
from utils import load_kb,get_tail_kb,get_attrname2entities
from utils import ch2pyinstr,is_chinese_str
import jieba
from collections import defaultdict


# In[2]:


def load_test_data(test_file):
    with open(test_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    samples = {}
    all_messages = []
    for i in range(len(data)):
        index = str(i+1)
        if len(data[index])==0:
            break
        question = data[index][-1].get("message")
        context = [turn["message"] for turn in data[index]]
        all_messages.extend(context)
        sample = {"question": question, "context": context}
        samples[index] = sample
    return samples


# In[3]:


# kb, entity_mapping = load_kb('../data/preliminary/kg.json')
# tail_kb = get_tail_kb(kb, entity_mapping)
# attrname2entities = get_attrname2entities(kb)


class Text_Correct():
    def __init__(self,):
        correct_attrnames = ['什么意思',
                    '英语翻译',
                    '英文翻译',
                    '外文翻译',
                    '思想感情',
                    '思想情感',
                    '什么样',
                    '出版社',
                    '代表作',
                    ]
        self.pyin2attrname = defaultdict(list)
        for x in correct_attrnames:   # attrname2entities.keys()
            if is_chinese_str(x) and len(x)>=2:
                self.pyin2attrname[ch2pyinstr(x)].append(x)
    
    def run(self, question):
        pyin2attrname = self.pyin2attrname
        new_question = deepcopy(question)
        question_str = ch2pyinstr(question)
        for x in pyin2attrname:
            if   x in question_str and pyin2attrname[x][0] not in question: # len(pyin2attrname[x][0])>=3
                print(f'question: {question}  {pyin2attrname[x]}')
                true_attrname =  pyin2attrname[x][0]
                attrname_len = len(true_attrname)
                false_name = None
                for j in range(len(question)):
                    t = question[j:j+attrname_len]
                    if ch2pyinstr(t)==x:
                        false_name = t
                if false_name is not None:
                    new_question = new_question.replace(false_name, true_attrname)
        return new_question


test_data = load_test_data('../data/preliminary/test.json')

# In[6]:


### 分词后纠正，纠错的比纠正的更多
# for i in range(len(test_data)):
#     key = str(i+1)
#     question = test_data[key]["question"]
#     question_tokens = jieba.lcut(question)
#     for j in range(len(question_tokens)):
#         x = question_tokens[j]
#         if is_chinese_str(x) and len(x)>=2:
#             x_str = ch2pyinstr(x)
#             if x_str in pyin2attrname and x not in pyin2attrname[x_str]:
#                 print(f"quesiton: {question} {x} {pyin2attrname[x_str]}")


# In[7]:


### 待膘咗->代表作
text_correcter = Text_Correct()

for i in range(len(test_data)):
    key = str(i+1)
    # if len(test_data[key])==0:
    #     break
    question = test_data[key]["question"]
    context = test_data[key]["context"]
    question = question.replace("。","")
    # question_str = ch2pyinstr(question)
    
    new_question = text_correcter.run(question)
    context[-1] = new_question
    # for x in pyin2attrname:
    #     if   x in question_str and pyin2attrname[x][0] not in question: # len(pyin2attrname[x][0])>=3
    #         print(f'question: {question}  {pyin2attrname[x]}')
    #         true_attrname =  pyin2attrname[x][0]
    #         attrname_len = len(true_attrname)
    #         false_name = None
    #         for j in range(len(question)):
    #             t = question[j:j+attrname_len]
    #             if ch2pyinstr(t)==x:
    #                 false_name = t
    #         if false_name is not None:
    #             context[-1] = question.replace(false_name, true_attrname)


# In[8]:


# mp = {}
# for i in range(400):
#     key = str(i+1)
#     context = test_data[key]["context"]
#     mp[key] = [{"message":x} for x in context]


# # In[9]:


# output_file = '../data/test_correct.json'
# with open(output_file,'w', encoding="UTF-8") as f:
#         json.dump(mp, f, ensure_ascii=False)

