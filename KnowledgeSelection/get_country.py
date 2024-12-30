#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
from itertools import chain
from collections import defaultdict
import re
from utils import load_kb, get_tail_kb
from utils import get_attrname2entities
random.seed(1)

kb, entity_mapping = load_kb('../data/preliminary/kg.json')
tail_kb = get_tail_kb(kb, entity_mapping)
attrname2entities = get_attrname2entities(kb)

a = attrname2entities['主要国家']
b = attrname2entities['主要城市']
c = attrname2entities['政治体制']
d = attrname2entities['司令部'] + attrname2entities['领导者姓名']

country = []
country.extend(a)
country.extend(b)
country.extend(c)
country.extend(d)
country = set(country)
country = sorted(list(country))
print('country size:', len(country))

# In[2]:


outputfile = './country.json'
with open(outputfile, 'w', encoding='utf-8') as f:
    json.dump(country, f, ensure_ascii=False)


# In[ ]:




