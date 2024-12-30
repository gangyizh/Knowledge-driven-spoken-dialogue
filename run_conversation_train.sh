python="/data/zhousf/anaconda3/envs/kdconv2/bin/python"

cd Generation
nohup bash train.sh >train.out
${python} cp_model.py