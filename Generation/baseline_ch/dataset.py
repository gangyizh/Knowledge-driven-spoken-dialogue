import os
import json
import random
import logging
import sys


from itertools import chain

import torch

from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)

try:
    from scripts.dataset_walker import DatasetWalker   ### train
except:
    from ..scripts.dataset_walker import DatasetWalker   ### infer

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"],
}
SPECIAL_TOKENS_VALUES = ["<bos>", "<eos>", "<pad>", "<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>"]

def change_knowledge_format(knowledge):
    attrname2items = {}
    for item in knowledge:
        attrname = item["attrname"]
        if attrname not in attrname2items:
            attrname2items[attrname] = []
        attrname2items[attrname].append(item)

    new_konws = []

    for attrname,items in attrname2items.items():
        name = items[0]['name']
        attrname = items[0]['attrname']
        attrvalue = []
        for item in items:
            attrvalue.append(str(item['attrvalue']))
        attrvalue = '、'.join(attrvalue)
        mp = {"attrname":attrname, "attrvalue": attrvalue,"name": name}
        new_konws.append(mp)
    return new_konws

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])  # begin
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])  # end
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])  # pad
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]  # "<knowledge_sep>"

        self.dataset_walker = DatasetWalker(split_type, dataroot=self.dataroot)

        self._create_examples()

    def _create_examples(self):
        logger.info("Creating examples")
        self.examples = []
        for bindex, (history, response, knowledge, dialog_id) in enumerate(
                tqdm(self.dataset_walker, disable=self.args.local_rank not in [-1, 0])):
            # if bindex>100:
            #     break
            history_convert = []  # 存放 history_ids
            index_cnt = 0
            for index in range(0, len(history)):
                # if index>0:
                #     break              ### 只考虑问题，不考虑历史对话
                history_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(history[len(history) - index - 1]["text"]))  # history语句逆序
                index_cnt += len(history_ids)
                if index_cnt >= self.args.history_max_tokens:                            # 超出长度截断
                    red = self.args.history_max_tokens - (index_cnt - len(history_ids))
                    history_convert.append(history_ids[0:red])
                    break
                history_convert.append(history_ids)
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response))
            tokenized_gt_resp = tokenized_gt_resp[:self.args.resp_max_tokens]

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(history_convert, self.args.history_max_tokens)  # 不懂意义，感觉多余
            if len(knowledge) == 0:
                used_knowledge = []
            else:
                knowledge = change_knowledge_format(knowledge)
                str_knowledge = ""
                for item in knowledge:
                    temp = "-".join([str(item["attrname"]), str(item["attrvalue"]), str(item["name"])])
                    str_knowledge = ";".join([str_knowledge, temp])
                used_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_knowledge))
                used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]

            self.examples.append({
                "history": truncated_history,  #list(ids)
                "knowledge": used_knowledge,  # ids
                "response": tokenized_gt_resp, # ids
                "response_text": response,  # text
                "dialog_id": dialog_id      # int
            })

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        ### 改为bart t5形式
        sequence = [sequence[0]] + sequence_with_speaker
        # instance["input_ids"] = list(chain(*sequence))  ### list(list(ids)) to list(ids)
        instance["input_ids"] = list(chain(*(sequence[:-1])))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        # instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["lm_labels"] = sequence[-1][1:]
        instance["attention_mask"] = [1]*len(instance["input_ids"])
        return instance, sequence

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        attention_mask = [ins["attention_mask"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = torch.tensor(pad_ids(attention_mask, 0))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, attention_mask, token_type_ids, lm_labels






class ResponseGenerationEvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch






class BaseOneDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, mp):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])  # begin
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])  # end
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])  # pad
        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]  # "<knowledge_sep>"

        self._create_examples(mp)

    def _create_examples(self, mp):
        ## logger.info("Creating examples")
        self.examples = []

        history, response, knowledge, dialog_id = mp["history"], mp["response"], mp["knowledge"], mp["dialog_id"]
        history_convert = []  # 存放 history_ids
        index_cnt = 0
        for index in range(0, len(history)):
            # if index>0:
            #     break              ### 只考虑问题，不考虑历史对话
            history_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(history[len(history) - index - 1]["text"]))  # history语句逆序
            index_cnt += len(history_ids)
            if index_cnt >= self.args.history_max_tokens:                            # 超出长度截断
                red = self.args.history_max_tokens - (index_cnt - len(history_ids))
                history_convert.append(history_ids[0:red])
                break
            history_convert.append(history_ids)
        tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(response))
        tokenized_gt_resp = tokenized_gt_resp[:self.args.resp_max_tokens]

        # apply history threshold at an utterance-level (a large value can be used to nullify its effect)

        # perform token-level truncation of history from the left 
        truncated_history = truncate_sequences(history_convert, self.args.history_max_tokens)  # 不懂意义，感觉多余
        if len(knowledge) == 0:
            used_knowledge = []
        else:
            knowledge = change_knowledge_format(knowledge)
            str_knowledge = ""
            for item in knowledge:
                temp = "-".join([str(item["attrname"]), str(item["attrvalue"]), str(item["name"])])
                str_knowledge = ";".join([str_knowledge, temp])
            used_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(str_knowledge))
            used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]

        self.examples.append({
            "history": truncated_history,  #list(ids)
            "knowledge": used_knowledge,  # ids
            "response": tokenized_gt_resp, # ids
            "response_text": response,  # text
            "dialog_id": dialog_id      # int
        })

    def build_input_from_segments(self, knowledge, history, response, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}

        sequence = [[self.bos] + knowledge] + history + [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        ### 改为bart t5形式
        sequence = [sequence[0]] + sequence_with_speaker
        # instance["input_ids"] = list(chain(*sequence))  ### list(list(ids)) to list(ids)
        instance["input_ids"] = list(chain(*(sequence[:-1])))
        instance["token_type_ids"] = [self.speaker2 if i % 2 else self.speaker1 for i, s in enumerate(sequence) for _ in s]
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        # instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["lm_labels"] = sequence[-1][1:]
        instance["attention_mask"] = [1]*len(instance["input_ids"])
        return instance, sequence

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class ResponseOneGenerationDataset(BaseOneDataset):
    def __init__(self, args, tokenizer, mp):
        super(ResponseOneGenerationDataset, self).__init__(args, tokenizer, mp)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch

