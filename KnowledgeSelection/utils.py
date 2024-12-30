import json
from webbrowser import get
from rank_bm25 import BM25Okapi
import torch
import os
import random
import numpy as np
import torch
import torch.nn as nn 
from collections import Counter,defaultdict
import re
import string
from time import strptime
from time import strftime
from pypinyin import lazy_pinyin,pinyin


attrname2inputname = {
    "Information":"简介",
    "开放时间":"开放什么游玩时间",
    "建议游玩时间":"参观游玩多久时长",
    "诗词全文":"诗词全文第一句最后一句背诵",
}

inputname2synonyms = {
    "开放什么游玩时间": ["什么时间","什么时候", "开放时间"],
    "参观游玩多久时长": ["建议游玩时间"],
    "介绍":["特征"],
    "主要成就": ["有名气"],
}

modelname2inputname = {v:k for k,v in attrname2inputname.items()}

# inputname2synonyms = {
#     "开放什么游玩时间": ["什么时间","什么时候", "开放时间","几点"],
#     "参观游玩多久时长": ["建议游玩时间","多久"],
#     "介绍":["特征特点","模样","时候"],   ### 只和生物有关
#     "主要成就": ["有名气"],
#     "诗词全文第一句最后一句背诵":["诗词全文","全文","第一句","最后一句","背诵"],
#     "评分":["评价"],
#     "出生日期":["岁数", "年纪"],
#     "中心思想":["情感"],
#     "作品赏析":["点评","评价"],
#     "赏析":["点评","评价"],
#     # "简介":["生平"]
# }

special_enties = ['0', '1', '1040', '119', '13', '15', '16', '1701', '1906', '1941', '1952', '1965', '1999', '20', 
                  '2001年9月11日', '2002', '2006', '2008', '2012', '2046', '21', '224', '23', '25', '3', '300', '31', '35',
                  '42', '5', '52', '8', '80', '9', '90', 'O', 'W', 'w', '一', '兰', '句', '叶', '她', '家', '弟', '扇', '春', 
                  '杏', '枣', '柽', '梨', '榆', '榴', '槿', '满', '燕', '爱', '画', '竹', '羊', '芦', '花', '葵', '蚊', '蝉', 
                  '赢', '雨', '雪', '韭', '飘', '鸟', '？']


def transform_attrname2inputname(x):
    if x not in attrname2inputname:
        return x
    return attrname2inputname[x]

def transform_inputname2attrname(x):
    if x not in modelname2inputname:
        return x
    return modelname2inputname[x]

### kg
def load_kb(kbfile):
    kb = {}
    entity_mapping = {}
    with open(kbfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    for entity in data:
        if "（" in entity:
            new_entity = entity.split("（")[0]
            if new_entity not in entity_mapping:
                entity_mapping[new_entity] = []     # 简化后的entity to 原来的entity
            entity_mapping[new_entity].append(entity)    # 针对 小宇宙(概念)，小宇宙(专辑)

            if len(entity.split("（"))>2:           # 针对 数据结构与算法分析（C++版）（第二版）
                new_entity = ''.join(entity.split("（")[:2])
                if new_entity not in entity_mapping:
                    entity_mapping[new_entity] = []     
                entity_mapping[new_entity].append(entity)  

        kb[entity] = {}
        for attr in data.get(entity):
            head, rel, tail = attr
            rel = transform_attrname2inputname(rel)   # 转换为输入模型的attrname
            if rel not in kb.get(entity):
                kb.get(entity)[rel] = []
            if tail not in kb.get(entity)[rel]:
                kb.get(entity)[rel].append(str(tail))
    print(f"length of kb: {len(kb)}")
    return kb, entity_mapping


def load_filter_kb(kbfile):
    filter_entities = []
    with open('data/country.json') as f:
        country_word = json.load(f)
    with open('data/filter_words.json') as f:
        filter_word = json.load(f)
    filter_entities.extend(country_word)
    filter_entities.extend(filter_word)

    kb, entity_mapping = load_kb(kbfile)

    new_kb = {}
    new_entity_mapping = {}

    for head,attrs in kb.items():
        if head in filter_entities:
            continue
        new_kb[head] ={}
        for attr,tail in attrs.items():
            new_kb[head][attr] = tail
    for k,v in entity_mapping.items():
        if k in filter_entities:
            continue
        new_entity_mapping[k] = v
    
    return new_kb, new_entity_mapping


def kb_completion(kb, extra_kb):
    kb['真精器鱼'] = extra_kb['真精器鱼']
    return kb

def get_tail_kb(kb, entity_mapping):
    tail_entities = []
    for key,attrs in kb.items():
        for attr,tails in attrs.items():
            tail_entities.extend(tails)
    entity2cnt = Counter(tail_entities)

    head_entities = set(kb.keys()|entity_mapping.keys())
    tail_entities = list(set(tail_entities))
    # tail_node_len = [len(t) for t in tail_entities]
    se_tail_entities = [t for t in tail_entities if len(t)<=10 and len(t)>0  and t not in head_entities] # entity2cnt[t]>=2
    se_tail_entities = [t for t in se_tail_entities if len(re.findall("\d", t))==0] 
    print(f'tail entity count: {len(se_tail_entities)}')

    se_tail_entities = set(se_tail_entities)
    tail_kb = {}
    skip_attrname = ['周边景点','门票',"分类","出版社",'性别','在线播放平台',"性质","朝代","所处时代","创作年代",
                    '界','门','纲','目','科','属','是否具备经济价值','是否可以作为食物','是否具备观赏价值','是否有毒']

    for head,attrs in kb.items():
        for attr,tails in attrs.items():
            if attr in skip_attrname:
                continue
            for tail in tails:
                if tail in se_tail_entities:
                    if tail not in tail_kb:
                        tail_kb[tail] = {}
                    if attr not in tail_kb[tail]:
                        tail_kb[tail][attr] = []
                    tail_kb[tail][attr].append(head)
    
    new_tail_kb = {}
    for tail,attrs in tail_kb.items():
        for attr,heads in attrs.items():
            if(len(heads))>=2 or attr in ['作者']:
                if tail not in new_tail_kb:
                    new_tail_kb[tail] = {}
                new_tail_kb[tail][attr] = heads

    return new_tail_kb

def get_complete_tail_kb(kb, entity_mapping):
    tail_entities = []
    for key,attrs in kb.items():
        for attr,tails in attrs.items():
            tail_entities.extend(tails)
    entity2cnt = Counter(tail_entities)

    head_entities = set(kb.keys()|entity_mapping.keys())
    tail_entities = list(set(tail_entities))
    # tail_node_len = [len(t) for t in tail_entities]
    # se_tail_entities = [t for t in tail_entities if len(t)<=10 and len(t)>0 and entity2cnt[t]>=2]
    se_tail_entities = [t for t in tail_entities if (len(t)<=10 or  (len(t)<20 and len(re.sub('[(（][a-zA-Z]+[)）]|[(（][a-zA-Z]+', '', t))<8))  and len(t)>0 and entity2cnt[t]>=2]
    se_tail_entities = [t for t in se_tail_entities if len(re.findall("\d", t))==0]
    print(f'tail entity count: {len(se_tail_entities)}')

    se_tail_entities = set(se_tail_entities)
    tail_kb = {}
    for head,attrs in kb.items():
        for attr,tails in attrs.items():
            if attr in ['周边景点','门票',"分类"]:
                continue
            for tail in tails:
                if tail in se_tail_entities:
                    if tail not in tail_kb:
                        tail_kb[tail] = {}
                    if attr not in tail_kb[tail]:
                        tail_kb[tail][attr] = []
                    tail_kb[tail][attr].append(head)
    
    new_tail_kb = {}
    for tail,attrs in tail_kb.items():
        for attr,heads in attrs.items():
            if(len(heads))>=2:
                if tail not in new_tail_kb:
                    new_tail_kb[tail] = {}
                new_tail_kb[tail][attr] = heads

    return new_tail_kb

def attrs2str(kb, key):
    attrs = kb[key]
    attrnames = sorted(list(attrs.keys()))
    attrvalues = []
    for attr_name in attrnames:
        v = attrs[attr_name]
        v = [str(t) for t in v]
        attrvalues.append('|'.join(v))
    
    attrnames = ','.join(attrnames)
    attrvalues = ','.join(attrvalues)
    
    return attrnames,attrvalues


def attrs2str_synonyms(kb, key, inputname2synonyms):
    attrs = kb[key]
    attrnames = sorted(list(attrs.keys()))
    attrvalues = []
    synonyms_attrnames = []
    for attr_name in attrnames:
        if attr_name in inputname2synonyms:
            synonyms_attrnames.extend(inputname2synonyms[attr_name])

        v = attrs[attr_name]
        v = [str(t) for t in v]
        attrvalues.append('|'.join(v))
    
    attrnames.extend(synonyms_attrnames)
    attrnames = ','.join(attrnames)
    attrvalues = ','.join(attrvalues)
    
    return attrnames,attrvalues



def get_attrname2entities(kb):
    attrname2entities = {}
    for entity,attrs in kb.items():
        for attrname,attrvalue in attrs.items():
            if attrname not in attrname2entities:
                attrname2entities[attrname] = set()
            attrname2entities[attrname].add(entity)
    
    for attrname in attrname2entities:
        attrname2entities[attrname] = sorted(list(attrname2entities[attrname]))
    return attrname2entities

### set seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### focal loss
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * ((1 - pt) ** self.gamma) * target * torch.log(pt) - (1 - self.alpha) * (pt ** self.gamma) * (1 - target) * torch.log(1 - pt)

        # loss = - target*torch.log(pt)-(1 - self.alpha)*torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



### BM25
def get_bm25(kb, entity_mapping):
    kb_enitites = list(kb.keys())
    all_kb_entity = list(kb.keys()|entity_mapping.keys())

    tokenized_corpus = [w for w in all_kb_entity]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def get_bm25_match(bm25, entity_corpus, query, threshold=5):
    scores = bm25.get_scores(query)
    best_docs = sorted(list(zip(entity_corpus, scores)), key=lambda x: x[1], reverse=True)[:10]
    if best_docs[0][1]>best_docs[1][1]+ threshold: # 
        return best_docs[0][0]
    ### todo 未满足threshold条件，但字符在文本都找得到，加入
    return None


def is_char_in_sent(w, sent):
    flag = True
    for c in w:
        if c not in sent:
            flag = False
            break
    return flag

def is_cnt_char_in_sent(w, query, sent):
    cnt = 0
    for c in w:
        if c in sent:
            cnt +=1
    miss_threshold = (len(query)-1)//4 + 1
    n_miss = len(w) - cnt
    flag = True if n_miss<=miss_threshold else False
    return flag

class BM25_Macth(object):
    def __init__(self, kb, entity_mapping):
        self.kb = kb
        self.entity_mapping = entity_mapping
        all_kb_entity = list(kb.keys()|entity_mapping.keys())

        self.tokenized_corpus = [w for w in all_kb_entity]
        self.bm25 = self.get_bm25(kb, entity_mapping)
        self.cache_mp ={}

    def get_bm25(self, kb, entity_mapping):
        kb_enitites = list(kb.keys())
        all_kb_entity = list(kb.keys()|entity_mapping.keys())

        tokenized_corpus = [w for w in all_kb_entity]
        bm25 = BM25Okapi(tokenized_corpus)
        
        return bm25
    
    def get_bm25_match(self, query, sent, threshold=5):
        if query in self.cache_mp:
            return self.cache_mp[query]
        scores = self.bm25.get_scores(query)
        best_docs = sorted(list(zip(self.tokenized_corpus, scores)), key=lambda x: x[1], reverse=True)[:10]

        match = []
        ### 多字的情况
        for (w,s) in best_docs:
            if w in sent and len(w)<len(query):
                match.append(w)
                break
        
        ### 缺字的情况
        for (w,s) in best_docs:
            if is_cnt_char_in_sent(w, query, sent) and len(w)>len(query):
                match.append(w)
                break
        
        ### 其他情况
        if best_docs[0][1]>best_docs[1][1]+ threshold: # 
            match.append(best_docs[0][0])
        match = list(set(match))

        ### todo 未满足threshold条件，但字符在文本都找得到，加入
        self.cache_mp[query] = match
        return match
        


### 字符串处理

### query字符串处理
def pre_process_query_snet(query):
    # x = re.findall("表达(.+?)\作者",query)
    # if len(x)>0:
    #     query = query.replace("作者", "")
    if "表达" in query and "作者" in query:
        query = query.replace("作者", "")
    return query

def get_multi_query(query):
    query = query.replace("?","？")
    x = re.findall("(.+?)\？", query)
    multi_query = []
    for t in x:
        if len(t)>=4:
            multi_query.append(t+"？")
    return multi_query

### 实体字符串处理
def is_num_str(x):
    if x is None or len(x)==0:
        return True
    flag = True
    for t in x.split('.'):
        if not t.isdigit():
            flag = False
    return flag

def is_date_str(datestr):
    pattern = ('%Y年%m月%d日', '%Y-%m-%d', '%y年%m月%d日', '%y-%m-%d','%m月%d日','%Y年')
    for i in pattern:
        try:
            ret = strptime(datestr, i)
            if ret:
                return True
        except:
            continue
    return False

def is_entity_str(x):
    if x is None or len(x)==0:
        return False
    if is_num_str(x):
        return False
    if  is_date_str(x):
        return False
    return True

def is_entity_str_addtion(x):
    if x in special_enties:
        return True
    return is_entity_str(x)


def is_chinese_str(x):
    '''
    检查整个字符串是否包含中文字符串
    :param string: 需要检查的字符串
    :return: bool
    '''
    
    for ch in x:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def is_english_str(strs):
    '''
    判断字符串是否是英文单词
    '''
    for i in strs:
        if i not in string.ascii_lowercase + string.ascii_uppercase:
            return False
    return True


def text2chinese_str(x):
    '''
    将字符串过滤为中文字符串
    :param string: 输入的字符串
    '''
    
    ch_str = ''.join([ch for ch in x if u'\u4e00' <= ch <= u'\u9fff'])
    return ch_str



def digit2ch(x):
    x = x.replace("0","零")
    x = x.replace("1","一")
    x = x.replace("2","二")
    x = x.replace("3","三")
    x = x.replace("4","四")
    x = x.replace("5","五")
    x = x.replace("6","六")
    x = x.replace("7","七")
    x = x.replace("8","八")
    x = x.replace("9","九")
    return x


def ch2pyinstr(x):
    x = digit2ch(x)
    s = lazy_pinyin(x)
    s = '_'.join(s)
    return s

def get_pyin2ch(kb, entity_mapping, tail_kb):
    all_entities = list(kb.keys()|entity_mapping.keys()|tail_kb.keys())

    ch_entities = [t for t in all_entities if is_chinese_str(t)]

    pyin2ch = {}
    for t in ch_entities:
        x = ch2pyinstr(t)
        if x not in pyin2ch:
            pyin2ch[x] = []
        pyin2ch[x].append(t)
    return  pyin2ch


### 获取姓氏
def get_chxs_name(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    name = []
    for line in lines:
        for c in line:
            if is_chinese_str(c):
                name.append(c)
    return name

class CHP_NAME():
    def __init__(self, file_name='data/chxs.txt'):
        self.people_name = get_chxs_name(file_name)
        self.people_name_set = set(self.people_name)

    def is_people_name(self, name):
        if name is not None and len(name)>=2 and len(name)<=3 and name[0] in self.people_name_set:
            return True
        return False



### model
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

### soft prompt model
class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)



### 成语含义case
def get_cwd_bm25_model(kb, attrname2entities):
    cwd_list = attrname2entities["释义"]
    
    cwd_value_list = [kb[x]["释义"][0] for x in cwd_list]
    value2cwds = defaultdict(list)
    for x in cwd_list:
        value2cwds[kb[x]["释义"][0]].append(x)
    
    all_kb_entity = cwd_value_list
    tokenized_corpus = [w for w in all_kb_entity]

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, cwd_value_list, value2cwds

class CWDFinder():
    def __init__(self, kb):
        self.kb = kb
        self.attrname2entities = get_attrname2entities(kb)
        self.bm25, self.cwd_value_list, self.value2cwds = get_cwd_bm25_model(kb, self.attrname2entities)

    def run(self, query):
        x = re.findall("形容(.+?)\成语",query)
        if len(x)==0:
            x = re.findall("描述(.+?)\成语",query)
        if len(x)==0:
            x = re.findall("比喻(.+?)\成语",query)
        if len(x)==0:
            return None
        x = x[0]
        scores = self.bm25.get_scores(x)
        
        best_docs = sorted(list(zip(self.cwd_value_list, scores)), key=lambda x: x[1], reverse=True)[:10]
        select_value = best_docs[0][0]
        ans = []
        for x in self.value2cwds[select_value]:
            ans.append([x, '释义'])
        return ans


class QuestionMatch():
    def __init__(self, inputfile='../data/train.json'):
        self.inputfile = inputfile
        self.corpus_data = self.load_data(inputfile)
        self.pyinstr2attrs = self.get_pyinstr2attrs()

    def load_data(self, inputfile):
        with open(inputfile, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        corpus_data = []
        for sample in data:
            messages = sample.get("messages")
            previous_message = messages[0].get("message")
            
            for i in range(1, len(messages)):
                message = messages[i].get("message")
                if "attrs" in messages[i]:
                    question = previous_message
                    attrs = messages[i].get("attrs")
                    answer = message
                    qsample = dict(question=question, attrs=attrs, answer=answer,)
                    corpus_data.append(qsample)
                previous_message = message
        return corpus_data

    def get_pyinstr2attrs(self, ):
        pyinstr2attrs = {}
        for qsample in self.corpus_data:
            question = qsample['question']
            attrs = qsample['attrs']
            answer = qsample['answer']
            question_ch = text2chinese_str(question)
            if len(question)>=10:
                question_str = ch2pyinstr(question_ch)
                pyinstr2attrs[question_str] = attrs
        return  pyinstr2attrs
    
    def run(self, question):
        question_ch = text2chinese_str(question)
        if len(question)>=10:
            question_str = ch2pyinstr(question_ch)
            if question_str in self.pyinstr2attrs:
                return self.pyinstr2attrs[question_str]
        return None














