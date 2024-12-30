import json
import re
import random

random.seed(1)

# prefix_do_words = ["简单，","做法不是很难，","流程如下：","这个简单，",]
# prefix_poem_words = ["简单，","这个简单，","没问题，","难不倒我，",""]

# def get_prefix_word(prefix_words):
#     pre_word = ""
#     pre_word = random.sample(prefix_words, 1)[0]
#     return pre_word


def abc_post_process(mp):
    print("abc_post_process...")

    for idx in range(len(mp)):
        key = str(idx+1)
        knowledges = mp[key].get("attrs", [])
        message = mp[key].get("message", "")

        if len(knowledges)==1:
            attrname = knowledges[0]['attrname']
            if re.findall("拼音", attrname):
                attr_value = knowledges[0]['attrvalue']
                new_cur_conv_ans = re.sub(r'[a-zA-Z]+', attr_value, message)
                mp[key]['message'] = new_cur_conv_ans
                print("cur_id:", key)
                print("old ans:", message)
                print('new ans:', new_cur_conv_ans)
                print("")
            
            if re.findall("外文名", attrname):
                attr_value = knowledges[0]['attrvalue']
                new_cur_conv_ans = re.sub(r'[a-zA-Z,-]+', attr_value, message) 
                mp[key]['message'] = new_cur_conv_ans
                print("cur_id:", key)
                print("old ans:", message)
                print('new ans:', new_cur_conv_ans)
                print("")
            
    return mp

def information_insert_process(mp):
    print("information_insert_process...")

    for idx in range(len(mp)):
        key = str(idx+1)
        knowledges = mp[key].get("attrs", [])
        message = mp[key].get("message", "")

        if len(knowledges)>0:
            ### "作者简介"
            flag = 0
            if knowledges[0]['attrname'] in ["介绍","赏析","中心思想","作品简介",'作者简介'] : ### or len(str(knowledges[0]['attrvalue']))>=40
                new_cur_conv_ans = str(knowledges[0]['attrvalue'])
                mp[key]["message"] = new_cur_conv_ans
                flag = 1
            
            if knowledges[0]['attrname'] in ["诗词全文"] : ### or len(str(knowledges[0]['attrvalue']))>=40
                new_cur_conv_ans = str(knowledges[0]['attrvalue'])
                mp[key]["message"] = new_cur_conv_ans
                flag = 1

            if knowledges[0]['attrname'] in ["做法"] and len(message)>=30:
                new_cur_conv_ans = str(knowledges[0]['attrvalue'])
                mp[key]["message"] = new_cur_conv_ans
                flag = 1

            if flag:
                print("cur_id:", key)
                print("old ans:", message)
                print('new ans:', new_cur_conv_ans)
                print("")

    return mp

def get_unk_words(file):
    with open(file, 'r', encoding='utf-8') as f:
        unk_words = json.load(f)
    return unk_words

def rare_word_process(mp, unk_words):
    print("rare_word_process...")
    unk_pattern = "[" + "".join(unk_words) + "]"

    for idx in range(len(mp)):
        key = str(idx+1)
        knowledges = mp[key].get("attrs", [])
        message = mp[key].get("message", "")
        for attr in knowledges:
            if re.findall(unk_pattern, attr["attrvalue"]):
                remove_rare_word_pattern = re.sub(unk_pattern, "[\u4E00-\u9FA5]", attr["attrvalue"])
                new_cur_conv_ans = re.sub(remove_rare_word_pattern, attr["attrvalue"],
                                        message)
                if message != new_cur_conv_ans:
                    print("cur_id:", key)
                    print("attrvalue:", attr["attrvalue"])
                    print("old ans:", message)
                    print('new ans:', new_cur_conv_ans)
                    print("")
                    # 替换答案
                    mp[key]['message'] = new_cur_conv_ans
            if re.findall(unk_pattern, attr["name"]):
                attrvalue_str = attr["name"]
                if attrvalue_str[:-1] in message and attrvalue_str not in message:
                    new_cur_conv_ans = message.replace(attrvalue_str[:-1], attrvalue_str)
                    mp[key]['message'] = new_cur_conv_ans
    return mp


input_file = './test_sub_result.json'
with open(input_file, 'r', encoding='utf-8') as f:
    mp = json.load(f)

unk_words = get_unk_words('../data/unk_words.json')

mp = abc_post_process(mp)
mp = information_insert_process(mp)
mp = rare_word_process(mp, unk_words)

# for idx in range(len(mp)):
#     knowledges = mp[str(idx+1)].get("attrs", [])
#     if len(knowledges)>0:
#         if knowledges[0]['attrname'] in ["做法","Information","故事","作者简介","赏析"] : ### or len(str(knowledges[0]['attrvalue']))>=40
#             mp[str(idx+1)]["message"] = str(knowledges[0]['attrvalue'])

output_file = 'test_sub_post_result.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(mp, f, ensure_ascii=False)