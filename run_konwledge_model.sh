#!/bin/bash
python="/data/zhousf/anaconda3/envs/kdconv2/bin/python"
cd KnowledgeSelection/NameEntityRecognition
${python}  preprocess.py
${python}  run_model.py
cd ../IntentBinExtraction
${python} preprocess.py
nohup ${python} run_bin_intent.py  --pretrain_model_path  ../../pretrain_model/chinese-macbert-base/ --save_model_path ../model/intent_macbert_1  >run_macbert1.out
nohup ${python} run_bin_intent.py  --pretrain_model_path  ../../pretrain_model/ernie-1.0-base-zh/ --save_model_path ../model/intent_erniezh_1  >run_erniezh1.out
cd ../EntitySelection
${python} preprocess.py
${python} process_entity_data.py
${python} run_entity_select.py