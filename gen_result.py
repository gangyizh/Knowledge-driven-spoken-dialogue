import argparse
import sys
import os
import json
from tqdm import tqdm
from KnowledgeSelection_1.run_cb_ent_attr_inference_multi_attrname import KnowledgeSelection
from Generation.gen_reply import GenReply
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_knowledge_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,
                        default="test")

    parser.add_argument('--tag_file', type=str,
                        default="KnowledgeSelection_1/data/tag.txt")
    parser.add_argument('--ner_pretrain_model_path', type=str,
                        default="pretrain_model/chinese-roberta-wwm-ext")
    parser.add_argument('--ner_save_model_path', type=str,
                        default="KnowledgeSelection_1/model/ner/")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ner_max_seq_len', type=int, default=512)

    parser.add_argument('--ent_select_pretrain_model_path', type=str,
                        default="pretrain_model/ernie-3.0-base-zh/")
    parser.add_argument('--ent_select_save_model_path', type=str,
                        default="KnowledgeSelection_1/model/entity_select_ernie3/")

    parser.add_argument('--extractor_max_seq_len', type=int, default=50)
    parser.add_argument('--max_conv_seq_len', type=int, default=400)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--max_entity_len', type=int, default=40)  # 多实体
    parser.add_argument('--max_attrname_len', type=int, default=40)
    parser.add_argument('--max_attrvalue_len', type=int, default=40)

    parser.add_argument('--bm25_threshold', type=float, default=2)
    parser.add_argument('--kb_file', type=str,
                        default="data/final_data/new_kg.json")
    parser.add_argument('--valid_file', type=str,
                        default="data/final_data/Noise-added/valid.json")
    parser.add_argument('--test_file', type=str,
                        default="data/final_data/test.json")
    parser.add_argument('--result_file', type=str,
                        default="test_know_result.json")
    parser.add_argument('--val_result_file', type=str,
                        default="eval_know_result.json")

    parser.add_argument('--debug', type=bool,
                        default=False)

    know_args = parser.parse_args()

    return know_args


know_args = get_knowledge_args()
know_selector = KnowledgeSelection(know_args)
reply_generator = GenReply()


with open('data/final_data/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

result = {}
for i in tqdm(range(len(test_data))):
    key = str(i+1)
    history = [t["message"] for t in test_data[key]]
    if len(history)==0:
        break
    conv_input = know_selector.inference_konwledge(history)
    conv_output = reply_generator.exec(conv_input)
    conv_output["history"] = history
    result[key] = conv_output

with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)



