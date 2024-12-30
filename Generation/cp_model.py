import shutil
import os

input_dir = './runs/rg-hml128-kml128-bart-large-chinese-att_mask/'
dir_names = os.listdir(input_dir)

source_dir = None
target_dir = 'checkpoint-15800'
for x in dir_names:
    if target_dir in x and x!=source_dir:
        source_dir = x
        print("source_dir:", source_dir)
        
if source_dir is not None:
    source_dir = os.path.join(input_dir, source_dir)
    target_dir = os.path.join(input_dir, target_dir)
    print(f'copy {source_dir} to {target_dir}')
    shutil.copytree(source_dir, target_dir)
else:
    print("no source_dir")


