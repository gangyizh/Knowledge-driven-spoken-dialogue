import os
from transformers import AutoTokenizer,AutoConfig,AutoModel,AutoModelForCausalLM


model_name = 'HIT-TMG/dialogue-bart-large-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)

os.makedirs(model_name, exist_ok=True)
tokenizer.save_pretrained(model_name)

model.save_pretrained(model_name)

config.save_pretrained(model_name)