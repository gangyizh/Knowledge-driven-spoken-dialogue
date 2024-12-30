#!/bin/bash

version="bart-large-chinese-att_mask_hit_bart"
dataroot="data/"
num_gpus=1

export CUDA_VISIBLE_DEVICES='7'


if [ ${num_gpus} = 1 ]; then
  python="/data/zhousf/anaconda3/envs/kdconv2/bin/python3"
else
  python="/data/zhousf/anaconda3/envs/kdconv2/bin/python3 -m torch.distributed.launch --nproc_per_node ${num_gpus}"
fi

# Response generation
${python} baseline.py \
    --params_file baseline_ch/configs/generation/params_hit_bart.json \
    --dataroot ${dataroot} \
    --exp_name rg-hml128-kml128-${version}  \
    --my_seed 42