import json
import numpy as np
import copy
import re
# import zhconv

# def simple_ch(x):
#     return zhconv.convert(x, 'zh-hans')

def process_reverse_konwledge(attrs):
    new_attrs = []
    for x in attrs:
        x_c = copy.deepcopy(x)
        if "know_reverse" in x_c: 
            # print(x_c["know_reverse"])
            del x_c["know_reverse"]
            temp = x_c['name']
            x_c['name'] = x_c['attrvalue']
            x_c['attrvalue'] = temp
            print(x_c)
        new_attrs.append(x_c)
    return new_attrs

def process_konwledge(attrs):
    new_attrs = process_reverse_konwledge(attrs)
    # for x in new_attrs:
    #     x['name'] = simple_ch(x['name'])
    #     x['attrvalue'] = simple_ch(x['attrvalue'])
    return new_attrs


def get_conv_input(mp, key="0"):
    history = []
    for idx,message in enumerate(mp["context"]):
        if idx == len(mp["context"])-1:
            # message = message.replace("。","")
            # message = message.replace("，","")
            # message = message.replace("？","")
            
            # new_text = re.sub(r'[，。？]+', "", message[:-1]) + message[-1]
            new_text = re.sub(r'[。]+', "", message[:-1]) + message[-1]

            # new_text2 = re.sub(r'([\u4e00-\u62db\u62dd-\u8c21\u8c23\u9fa5])\1', r'\1', new_text)
            message = new_text
        if idx%2==0:
            history.append({"speaker": "U", "text":message})
        else:
            history.append({"speaker": "S", "text":message})
    attrs = mp.get("attrs",[])
    new_attrs = process_konwledge(attrs)
    qsample = dict(history=history, response="", knowledge=new_attrs, dialog_id=key)
    return qsample


with open('test_know_post_result.json', 'r', encoding='utf-8') as f:
    mp = json.load(f)

conv_data = []

for i in range(300):
    key = str(i+1)
    # for idx,message in enumerate(mp[key]["context"]):
    #     if idx == len(mp[key]["context"])-1:
    #         # message = message.replace("。","")
    #         # message = message.replace("，","")
    #         # message = message.replace("？","")
    #         new_text = re.sub(r'[，。？]+', "", message[:-1]) + message[-1]
    #         # new_text2 = re.sub(r'([\u4e00-\u62db\u62dd-\u8c21\u8c23\u9fa5])\1', r'\1', new_text)
    #         message = new_text
    #     if idx%2==0:
    #         history.append({"speaker": "U", "text":message})
    #     else:
    #         history.append({"speaker": "S", "text":message})
    # attrs = mp[key].get("attrs",[])
    # new_attrs = process_konwledge(attrs)
    # qsample = dict(history=history, response="", knowledge=new_attrs, dialog_id=key)
    qsample = get_conv_input(mp[key])
    qsample['dialog_id'] = key
    conv_data.append([qsample], key)

outputfile = 'data/test/data.json'
with open(outputfile, 'w', encoding='utf-8') as f:
    json.dump(conv_data, f, ensure_ascii=False)