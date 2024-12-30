#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import argparse


def load_data(inputfile, outputfile):
    with open(inputfile, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    pairs = []
    for sample in data:
        messages = sample.get("messages")
        context = []       # 所有对话
        entities = set()   # 所有实体
        for message in messages:
            context.append(message.get("message"))
            attrs = message.get("attrs", [])
            for attr in attrs:
                if isinstance(attr.get("name"), list):
                    entities.update(attr.get("name"))
                else:
                    entities.add(str(attr.get("name")))
        entities = sorted(list(entities))            ### 增加
        sequence_label(context, entities, pairs)

    with open(outputfile, 'w', encoding='utf-8') as fout:
        json.dump(pairs, fout, ensure_ascii=False)


def sequence_label(context, entities, pairs):
    for sequence in context:
        tags = ["O"] * len(sequence)  # BIOS序列
        sample = {"raw_text": sequence, "entity": {}}
        for entity in entities:
            if entity in sequence:
                start_index = get_sequence_labels(sequence, entity, tags)
                sample.get("entity")[entity] = start_index  # list

        if "B" in tags or "S" in tags:
            sample = {"text": sequence, "labels": tags}
            pairs.append(sample)


def get_sequence_labels(sequence, entity, tags):
    index = sequence.find(entity)
    start_index = []
    while (index != -1 and index + len(entity) < len(sequence) + 1):  # 修改  index + len(entity) < len(sequence) - 1
#         if index + len(entity) >= len(sequence) - 1:
#             print(sequence)
#             print(entity)

        if tags[index] == "O":
            if len(entity) == 1:
                tags[index] = "S"  # 单字 S single
            else:
                tags[index] = "B"  # 多字 B begin
            start_index.append(index)
        for i in range(index + 1, index + len(entity)):
            if tags[i] == "O":
                tags[i] = "I"     
        index = sequence.find(entity, index + len(entity))
    return start_index


# In[2]:


input_path = '../../data/final_data/Noise-free/'
output_path = '../data/'
train_file = os.path.join(input_path, "train.json")
output_train_file = os.path.join(output_path, "ner_train.json")
load_data(train_file, output_train_file)

input_path = '../../data/final_data/Noise-added/'
output_path = '../data/'
valid_file = os.path.join(input_path, "valid.json")
output_valid_file = os.path.join(output_path, "ner_valid.json")
load_data(valid_file, output_valid_file)

