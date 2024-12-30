#!/bin/bash

version="bart-large-chinese"

export CUDA_VISIBLE_DEVICES='0'
python="/data/zhousf/anaconda3/envs/kdconv2/bin/python3"

mkdir -p pred/val

${python} baseline.py \
        --generate  \
        --checkpoint "runs/rg-hml128-kml128-bart-large-chinese-att_mask_4/checkpoint-15800-val_loss_2.3226416450304255" \
        --generation_params_file baseline_ch/configs/generation/generation_params.json \
        --eval_dataset test \
        --dataroot data/ \
        --output_file test_evl.json
