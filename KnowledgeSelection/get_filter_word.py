#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from utils import load_kb, get_tail_kb, get_attrname2entities

filter_entities = ['华语','民族','乐团','是','制片人','学校','作家','漫画','语言','粤语','英语','时间','台湾','音乐风格','白骨再肉','台北','香港','政治',"土豆","土豆网",
                   '澳门','其他','男性','女性','成语','唐朝','明朝','民谣','诗人','艺术','美国人','我','成语','免费','喜剧片','动漫','演员','男演员','女演员','时间','人类',
                   '制作人','音乐人','制作人（职业名称）','魔兽世界',"北京","澳洲","动画片","无题（五首）"] 

kb, entity_mapping = load_kb('../data/preliminary/kg.json')
tail_kb = get_tail_kb(kb, entity_mapping)
attrname2entities = get_attrname2entities(kb)

A = [x for x in attrname2entities['拼音'] if len(x)<=2]
B = [x for x in attrname2entities['领域'] if x!='西屋太志']


filter_entities = filter_entities + A + B
### to do 游戏类型(完美世界), 播出频道




# In[2]:


outputfile = './filter_words.json'
with open(outputfile, 'w', encoding='utf-8') as f:
    json.dump(filter_entities, f, ensure_ascii=False)


# In[ ]:




