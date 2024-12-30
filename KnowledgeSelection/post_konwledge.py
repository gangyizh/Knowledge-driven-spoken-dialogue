import json
import re
from copy import copy, deepcopy
from .utils import load_kb,get_complete_tail_kb,kb_completion
from .utils import transform_inputname2attrname, transform_attrname2inputname
from .utils import QuestionMatch



def get_relation_attrs(kb, relations, entity):
    ### to do,返回直接对象
    new_attrs = []
    pre_entities = [entity]
    target_entities = []
    for x in relations:
        new_entities = []
        level_attrs = []
        for ent_name in pre_entities:
            if ent_name in kb and x in kb[ent_name]:
                for attrvalue in kb[ent_name][x]:
                    new_attrs.append({"name": ent_name, "attrname": x,"attrvalue":attrvalue})
                    level_attrs.append({"name": ent_name, "attrname": x,"attrvalue":attrvalue})
                    new_entities.append(attrvalue)
        pre_entities = new_entities
        target_entities = level_attrs
        
    return new_attrs

def drop_duplicate_know(attrs):
    '''
    attrs: list(dict())
    return duplicate attrs
    '''
    new_attrs = []
    key_str_set = set()
    for know in attrs:
        entity = know["name"]
        attrname = know["attrname"]
        attrvalue = know["attrvalue"]
        key_str = str(entity)+'_'+str(attrname)+'_'+str(attrvalue)
        if key_str not in key_str_set:
            key_str_set.add(key_str)
            new_attrs.append(know)
    return new_attrs


class PostKnowledge():
    def __init__(self,):
        self.kb, self.entity_mapping = load_kb('data/final_data/new_kg.json')
        # extra_kb, extra_entity_mapping = load_kb('../data/Comparison_Data/Knowledge_Graph_Data/Marine_Fishes/Modified_Data/fish_final_v2_16.json')
        # kb = kb_completion(kb, extra_kb)
        self.tail_kb = get_complete_tail_kb(self.kb, self.entity_mapping)
        # self.qa_macth = QuestionMatch(inputfile='../data/preliminary/train.json')

        wow_path = 'data/Knowledge_Graph_Data/Characters in World of Warcraft/Modified Data/wow_final.json'
        self.wow_kb,_ = load_kb(wow_path)

        ring_path = 'data/Knowledge_Graph_Data/The Lord of the Rings/Modified Data/the_ring_final.json'
        self.ring_kb,_ = load_kb(ring_path)

        fish_path = 'data/Knowledge_Graph_Data/Marine Fishes/Modified Data/fish_final_v2.json'
        self.fish_kb, _ = load_kb(fish_path)

    def run(self, know):
        ''' 
        know dict {question, attrs}
        return post process knowledge
        '''
        kb = self.kb
        tail_kb = self.tail_kb
        new_konw = deepcopy(know)

        question = know['question']
        attrs = know['attrs']
        
        question = re.sub(r'[，。？]+', "", question[:-1]) + question[-1]
        
        if len(attrs)>0:
            # print(attrs)
            entity = attrs[0]['name']
            attrname = attrs[0]['attrname']
            tail = attrs[0]['attrvalue']
            
            question = question.replace(entity, "")
            ent_attrnames = self.kb[entity].keys()
            
            # match_attrs = self.qa_macth.run(question)
            # if match_attrs is not None:
            #     match_attrname = match_attrs[0]['attrname']
            #     pred_attr_names = [x['attrname'] for x in attrs]
            #     if match_attrname not in pred_attr_names:
            #         new_attrs = []
            #         for x in ent_attrnames:
            #             trans_attrname = transform_inputname2attrname(x)
            #             if x==match_attrname:
            #                 add_attrs = [{"name": entity,"attrname": trans_attrname,"attrvalue":v,} for v in kb[entity][x]]
            #                 new_attrs.extend(add_attrs)
            #         if len(new_attrs)>0:
            #             print(question)
            #             print(attrs)
            #             print(new_attrs)
            #             attrs = new_attrs
            
            
            
            ### 生物学相关
            sw_class = ['界','门','纲','目','科','属']
            if attrname in sw_class and len(attrs)==1:
                for x in ent_attrnames:
                    if x in question and x!=attrname:
                        if (x=='属' and '属于' in question):
                            continue
                        x = transform_inputname2attrname(x)
                        add_attrs = [{"name": entity,"attrname": x,"attrvalue":v,} for v in kb[entity][x]]
                        attrs.extend(add_attrs)
                        
            if  '学类' in question or '分类' in question and '介绍' in kb[entity][x]:  ### 生物学类
                flag = True
                for x in sw_class:
                    if (x=='属' and '属于' in question):
                        continue
                    if x in question:
                        flag = False
                if flag:
                    attrs = []
                    for x in sw_class:
                        if x in kb[entity]:
                            add_attrs = [{"name": entity,"attrname": x,"attrvalue":v,} for v in kb[entity][x]]
                            attrs.extend(add_attrs)
            
            ### 父母
            if (attrname=='父亲' or attrname=='母亲') and '父' in question and '母' in question:
                attrs = []
                for x in ['父亲', '母亲']:
                    add_attrs = [{"name": entity,"attrname": x,"attrvalue":v,} for v in kb[entity][x]]
                    attrs.extend(add_attrs)
                    
            ### aa（bb）
            attrname = attrname.replace("（","(")
            attrname = attrname.replace("）",")")
            if '(' in attrname and ')' in attrname:
                a = attrname.split('(')[0]
                b = attrname.split('(')[-1].split(")")[0]
                if a in question and b not in question:
                    attrs = []
                    for x in ent_attrnames:
                        if a in x:
                            add_attrs = [{"name": entity,"attrname": x,"attrvalue":v,} for v in kb[entity][x]]
                            attrs.extend(add_attrs)
                            
            ### 共同xx 
            # if  "共同" in question or "同一" in question  or "一样" in question:
            if  "同" in question or "一样" in question:
                if attrname in question and tail in self.tail_kb and attrname in self.tail_kb[tail]:
                    new_attrs = [{"name": v,"attrname": attrname,"attrvalue":tail, "know_reverse":1} for v in self.tail_kb[tail][attrname] if v!=entity]
                    if len(new_attrs)>0 and len(new_attrs)<=50:
                        attrs.extend(new_attrs)
                        attrs = attrs[::-1]
                    elif len(new_attrs)>50:
                        new_attrs = new_attrs[:10]
                        attrs.extend(new_attrs)
                        attrs = attrs[::-1]
            
            ### 指环王 家族关系
            relation_attrnames = ["父亲","母亲","配偶","子嗣","兄弟姐妹"]
            if entity in self.ring_kb.keys():
                # if attrname in relation_attrnames:
                final_attrs = []
                 ### 孙子
                if '孙' in question:
                    relations = ["子嗣","子嗣"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "太爷" in question or "曾祖父" in question:
                    relations = ["父亲","父亲", "父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "太奶" in question or "曾祖母" in question:
                    relations = ["父亲","父亲", "母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "爷" in question and "太" not in question or ("祖父" in question and "曾" not in question):
                    relations = ["父亲","父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "奶" in question and "太" not in question or ("祖母" in question and "曾" not in question):
                    relations = ["父亲","母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "外公" in question or "外祖父" in question:
                    relations = ["母亲","父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "外婆" in question and "外祖母" in question:
                    relations = ["母亲","母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "叔" in question or "伯" in question or "父辈" in question or "姑" in question:
                    relations = ["父亲","兄弟姐妹"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "公公" in question:
                    relations = ["配偶","父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "婆婆" in question:
                    relations = ["配偶","母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "老婆" in question and "娘家" in question:
                    relations = ["配偶","母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                    relations = ["配偶","父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                
                if "母亲" in question and "娘家" in question:
                    relations = ["母亲","母亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                    relations = ["母亲","父亲"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                ### 侄子
                if "侄" in question or (('兄' in question or '弟' in question or'姐' in question or'妹' in question) 
                and ('儿子' in question or '孩' in question or '后代' in question)):
                    relations = ["兄弟姐妹","子嗣"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "弟媳" in question or "弟妹" in question or "嫂" in question or "连襟" in question:
                    relations = ["兄弟姐妹","配偶"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)

                if "儿媳" in question or "女婿" in question or ("孩" in question and "结婚" in question) or ("孩" in question and "成家" in question):
                    relations = ["子嗣","配偶"]
                    new_attrs = get_relation_attrs(kb, relations, entity)
                    final_attrs.extend(new_attrs)
                if len(final_attrs)>0:
                    final_attrs = drop_duplicate_know(final_attrs)
                    attrs = final_attrs
                
            
            
                
            ### 针对魔兽世界话题，添加question出现的属性, 添加联盟部落态度
            # if '职业' in ent_attrnames and '阵营' in ent_attrnames: ###判断魔兽世界人物 魔兽世界
            if entity in self.wow_kb.keys():
                ### 添加question出现的属性
                for x in ent_attrnames:
                    if x in question and x!=attrname and len(x)>=2:

                        x = transform_inputname2attrname(x)
                        add_attrs = [{"name": entity,"attrname": x,"attrvalue":v} for v in kb[entity][x]]
                        attrs.extend(add_attrs)
                
                ### 添加联盟部落态度
                if "联盟" in question and ("玩家" in question or "态度" in question):
                    x = "对联盟玩家态度"
                    attrnames = [t['attrname'] for t in attrs]
                    if x not in attrnames and x in kb[entity]:
                        add_attrs = [{"name": entity,"attrname": x,"attrvalue":v} for v in kb[entity][x]]
                        attrs.extend(add_attrs)
                if "部落" in question and ("玩家" in question or "态度" in question):
                    x = "对部落玩家态度"
                    attrnames = [t['attrname'] for t in attrs]
                    if x not in attrnames and x in kb[entity]:
                        add_attrs = [{"name": entity,"attrname": x,"attrvalue":v} for v in kb[entity][x]]
                        attrs.extend(add_attrs)
                
                ### 添加属性
                if "属性" in question:
                    new_attrs = []
                    select_attrnames = ["法力值","生命值","等级"]
                    for x in select_attrnames:
                        if x in kb[entity]:
                            add_attrs = [{"name": entity,"attrname": x,"attrvalue":v} for v in kb[entity][x]]
                            new_attrs.extend(add_attrs)
                    attrs = new_attrs  
                
                
                    
                
        know['attrs'] = attrs
        return know
































