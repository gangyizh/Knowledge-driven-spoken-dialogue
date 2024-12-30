import json
import copy

def simplify_attrs(attrs):
    new_attrs = []
    for x in attrs:
        x_c = copy.deepcopy(x)
        if "know_reverse" in x_c:
            del x_c["know_reverse"]
        new_attrs.append(x_c)
    return new_attrs


input_file = './test_evl.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('test_know_post_result.json', 'r', encoding='utf-8') as f:
    konwledge_data = json.load(f)

n_samaple = len(konwledge_data)
sub_data = {}
for i in range(n_samaple):
    key = i
    # attrs = data[key].get('knowledge', [])

    attrs = konwledge_data[str(i+1)].get("attrs",[])
    attrs = simplify_attrs(attrs)

    answer = data[key].get('generated_response', "")
    if len(attrs)==0:
        line_data = dict(message=answer)
    else:
        line_data = dict(attrs=attrs, message=answer)
    sub_data[str(i+1)] = line_data


### 多余，应对数据集的
for i in range(300,400):
    sub_data[str(i+1)] = {}

output_file = "./test_sub_result.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sub_data, f, ensure_ascii=False)

    







