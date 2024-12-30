import argparse
import logging
import os
import random
import json
import re
import copy
from typing import Dict
from argparse import Namespace

import numpy as np
import torch

from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    GPT2LMHeadModel,
    BartForConditionalGeneration,
    XLNetLMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from .baseline_ch.dataset import ResponseGenerationEvalDataset, ResponseOneGenerationDataset
from .baseline_ch.utils.argument import (
    update_additional_params,
)
from .baseline_ch.utils.model import run_batch_generation_sample, run_batch_generation_seq2seq_sample
# from .utils.metrics import (
#     UnigramMetric, NGramDiversity,
#     CorpusNGramDiversity,
#     BLEU
# )
# from .utils.data import write_generation_preds
# from .modeling_cpt import CPTModel, CPTForConditionalGeneration

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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

# def fish_value_process(attrs):
#     new_attrs = []
#     need_attrname = ['是否具备经济价值','是否可以作为食物','是否具备观赏价值','是否有毒']
#     for x in attrs:
#         x_c = copy.deepcopy(x)
#         attrname = x_c['attrname']
#         if attrname in need_attrname:
#             attrvalue = x_c['attrvalue']
#             if attrvalue == '是':
#                 x_c['attrvalue'] = "是 具备 可以"
#             else:
#                 x_c['attrvalue'] = "否 不具备 不可以"
#         new_attrs.append(x_c)
#     return new_attrs


def process_konwledge(attrs):
    new_attrs = process_reverse_konwledge(attrs)
    # new_attrs = fish_value_process(new_attrs)
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


def get_unk_words(file):
    with open(file, 'r', encoding='utf-8') as f:
        unk_words = json.load(f)
    return unk_words

def abc_post_process(mp):
    # print("abc_post_process...")

    knowledges = mp.get("attrs", [])
    message = mp.get("message", "")

    if len(knowledges)==1:
        attrname = knowledges[0]['attrname']
        if re.findall("拼音", attrname):
            attr_value = knowledges[0]['attrvalue']
            new_cur_conv_ans = re.sub(r'[a-zA-Z]+', attr_value, message)
            mp['message'] = new_cur_conv_ans
            # print("old ans:", message)
            # print('new ans:', new_cur_conv_ans)
            # print("")
        
        if re.findall("外文名", attrname):
            attr_value = knowledges[0]['attrvalue']
            new_cur_conv_ans = re.sub(r'[a-zA-Z,-]+', attr_value, message) 
            mp['message'] = new_cur_conv_ans
            # print("old ans:", message)
            # print('new ans:', new_cur_conv_ans)
            # print("")
            
    return mp

def abc_post_process_complete(mp):
    # print("abc_post_process...")

    knowledges = mp.get("attrs", [])
    message = mp.get("message", "")

    if len(knowledges)==1:
        attrname = knowledges[0]['attrname']
        if re.findall("拼音", attrname):
            attr_value = knowledges[0]['attrvalue']
            new_cur_conv_ans = re.sub(r'[a-zA-Z]+', attr_value, message)
            mp['message'] = new_cur_conv_ans
            # print("old ans:", message)
            # print('new ans:', new_cur_conv_ans)
            # print("")
            return mp
        
        attr_value = knowledges[0]['attrvalue']
        attr_value_clean = re.sub(r'[ ,-.]+', '', attr_value).lower()
        remove_china_char = re.findall('[a-zA-Z\u00E0-\u02EF]+', attr_value_clean)
        if len(remove_china_char) > 0:
            remove_char = remove_china_char[0]
            # print("remove_china_char:{} , attr_value_clean:{}".format(remove_china_char, attr_value_clean) )
            if remove_char == attr_value_clean:
                # print("attr_value_clean:", attr_value_clean)
                cur_conv_ans_clean =  re.sub(r'[-,.]+', '', message).lower()
                if len(re.findall(attr_value_clean, cur_conv_ans_clean)) > 0:
                    new_cur_conv_ans = re.sub(attr_value_clean, attr_value, cur_conv_ans_clean)
                    mp['message'] = new_cur_conv_ans
            
    return mp

def information_insert_process(mp):
    # print("information_insert_process...")


    knowledges = mp.get("attrs", [])
    message = mp.get("message", "")

    if len(knowledges)>0:
        ### "作者简介"
        flag = 0
        if knowledges[0]['attrname'] in ["介绍","赏析","中心思想","作品简介",'作者简介'] : ### or len(str(knowledges[0]['attrvalue']))>=40
            new_cur_conv_ans = str(knowledges[0]['attrvalue'])
            mp["message"] = new_cur_conv_ans
            flag = 1
        
        if knowledges[0]['attrname'] in ["诗词全文"] : ### or len(str(knowledges[0]['attrvalue']))>=40
            new_cur_conv_ans = str(knowledges[0]['attrvalue'])
            mp["message"] = new_cur_conv_ans
            flag = 1

        if knowledges[0]['attrname'] in ["做法"] and len(message)>=30:
            new_cur_conv_ans = str(knowledges[0]['attrvalue'])
            mp["message"] = new_cur_conv_ans
            flag = 1

        # if flag:
        #     print("old ans:", message)
        #     print('new ans:', new_cur_conv_ans)
        #     print("")

    return mp

def rare_word_process(mp, unk_words):
    # print("rare_word_process...")
    unk_pattern = "[" + "".join(unk_words) + "]"


    knowledges = mp.get("attrs", [])
    message = mp.get("message", "")
    for attr in knowledges:
        if re.findall(unk_pattern, attr["attrvalue"]):
            remove_rare_word_pattern = re.sub(unk_pattern, "[\u4E00-\u9FA5]", attr["attrvalue"])
            new_cur_conv_ans = re.sub(remove_rare_word_pattern, attr["attrvalue"],
                                    message)
            if message != new_cur_conv_ans:
                print("attrvalue:", attr["attrvalue"])
                print("old ans:", message)
                print('new ans:', new_cur_conv_ans)
                print("")
                # 替换答案
                mp['message'] = new_cur_conv_ans
        # if re.findall(unk_pattern, attr["name"]):
        #     attrvalue_str = attr["name"]
        #     if attrvalue_str[:-1] in message and attrvalue_str not in message:
        #         new_cur_conv_ans = message.replace(attrvalue_str[:-1], attrvalue_str)
        #         mp[key]['message'] = new_cur_conv_ans
    return mp

def rare_word_process2(mp, unk_words):
    # print("rare_word_process...")
    unk_pattern = "[" + "".join(unk_words) + "]"


    cur_conv_ans = mp['message']
    cur_conv_attrs = mp['attrs']
    for attr in cur_conv_attrs:
        if re.findall(unk_pattern, attr["attrvalue"]):
            remove_rare_word_pattern = re.sub(unk_pattern, "[\u4E00-\u9FA5]", attr["attrvalue"]).replace('(', '\(').replace(')', '\)')
            # print('remove_rare_word_pattern', remove_rare_word_pattern)
            if len(re.findall(cur_conv_ans, attr["attrvalue"] )) > 0 and len(cur_conv_ans)<10:  # answer属于attrvalue子集
                new_cur_conv_ans =  attr["attrvalue"]  # answer替换
            else:  # 词语匹配
                new_cur_conv_ans = re.sub(remove_rare_word_pattern, attr["attrvalue"],
                                            cur_conv_ans)  # 匹配成功，替换
            if cur_conv_ans != new_cur_conv_ans:
                # print("cur_id:", conv_ind)
                # print("question:", question)
                # print("attrvalue:", attr["attrvalue"])
                # print("old ans:", cur_conv_ans)
                # print('new ans:', new_cur_conv_ans)
                # 替换答案
                mp['message'] = new_cur_conv_ans
        if re.findall(unk_pattern, attr["attrname"]):
            remove_rare_word_pattern = re.sub(unk_pattern, "[\u4E00-\u9FA5]", attr["attrname"]).replace('(', '\(').replace(')', '\)')
            if len(re.findall(cur_conv_ans, attr["attrname"])) > 0 and len(cur_conv_ans) < 10:  # answer属于attrvalue子集
                new_cur_conv_ans = attr["attrvalue"]  # answer替换
            else:
                new_cur_conv_ans = re.sub(remove_rare_word_pattern, attr["attrname"], cur_conv_ans)
                
            if cur_conv_ans != new_cur_conv_ans:
                # print("attrname:", attr["attrname"])
                # print("old ans:", cur_conv_ans)
                # print('new ans:', new_cur_conv_ans)
                # 替换答案
                mp['message'] = new_cur_conv_ans
    return mp


class GenReply():
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Required parameters
        parser.add_argument("--checkpoint", type=str,
                            default="Generation/runs/rg-hml128-kml128-bart-large-chinese-att_mask",
                            help="Saved checkpoint directory")  ### 路径修改
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="Device (cuda or cpu)")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="Local rank for distributed training (-1: not distributed)")
        args = parser.parse_args()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )

        # load args from params file and update the args Namespace
        args.params_file = os.path.join(args.checkpoint, "params.json")
        with open(args.params_file, "r") as f:
            params = json.load(f)
            args = vars(args)
            update_additional_params(params, args)
            args.update(params)
        # if len(args["generation_params_file"]) > 0:
        #     with open(args["generation_params_file"]) as fg:
        #         generation_params = json.load(fg)
        generation_params = { 
                            "no_sample": False,
                            "min_length": 1,
                            "max_length": 200,
                            "temperature": 0.7,
                            "top_k": 50,
                            "top_p": 0.6
                            }
        args.update(generation_params)
        args = Namespace(**args)

        args.params = params  # used for saving checkpoints
        dataset_args = Namespace(**args.dataset_args)
        dataset_args.local_rank = args.local_rank
        dataset_args.task = args.task

        # Setup CUDA, GPU & distributed training
        args.distributed = (args.local_rank != -1)
        if not args.distributed:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method='env://')
        args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
        args.device = device
        # Set seed
        set_seed(args)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

        args.output_dir = args.checkpoint
        # tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
        # model = GPT2LMHeadModel.from_pretrained(args.checkpoint)

        tokenizer = BertTokenizer.from_pretrained(args.checkpoint)
        ### to do: 加入多余的编码字
        unk_words_file = 'data/unk_words.json'  ### 路径修改
        unk_words = get_unk_words(unk_words_file)
        tokenizer.add_tokens(unk_words)
        model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
        model.to(args.device)
        model.resize_token_embeddings(len(tokenizer))

        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

        logger.info("Generation parameters %s", args)

        self.args = args
        self.dataset_args = dataset_args
        # self.split_type=args.eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.unk_words = unk_words
        self.model.eval()

    def exec(self, mp):
        conv_input = get_conv_input(mp)   ### 转换为对话输入形式

        args = self.args
        dataset_args = self.dataset_args
        # args.eval_dataset = self.split_type
        tokenizer = self.tokenizer
        model = self.model

        args.eval_batch_size = 1

        eval_dataset = ResponseOneGenerationDataset(dataset_args, tokenizer, conv_input)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=1,  # only support batch_size=1 for sampling right now
            collate_fn=eval_dataset.collate_fn
        )

        args.tokenizer = tokenizer
        all_output_texts = []
        # self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            # print(batch)
            with torch.no_grad():
                sampled_output_ids, ground_truth, dialog_id = run_batch_generation_seq2seq_sample(args, model, batch, eval_dataset)
                sampled_output_ids = sampled_output_ids.squeeze(0)
                sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
                all_output_texts.append(sampled_output_text)
        r = all_output_texts[0]
        r = r.replace(" ", "")
        conv_output = {"attrs":mp["attrs"],"message":r}  ### 需要 del know_reverse
        try:
            conv_output = abc_post_process_complete(conv_output)
            conv_output = information_insert_process(conv_output)
            conv_output = rare_word_process2(conv_output, self.unk_words)
        except :
            print("post process error")
        for t in conv_output["attrs"]:
            if "know_reverse" in t:
                del t["know_reverse"]
        return conv_output


if __name__ == "__main__":
    # main()
    with open('test_know_post_result.json', 'r', encoding='utf-8') as f:
        mp = json.load(f)
    gen_model = GenReply()
    messages = []
    
    for i in tqdm(range(300)):
        key = str(i+1)
        conv_output = gen_model.exec(mp[key])    
        messages.append(conv_output)

    with open('test_message.json', 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False)
    
