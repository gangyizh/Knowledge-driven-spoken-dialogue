import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModel,AutoConfig,BertTokenizer
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from entselect_model import ExtractorModel
from entselect_dataset import DatasetExtractor

def truncate_str(x, m_len):
    '''
    文本超出范围则取前后部分
    '''
    if len(x)<=m_len:
        return x
    else:
        return x[:(m_len+1)//2] + x[-(m_len//2):]

class EntInfere():
    def __init__(self, gpu, pretrain_model_path, save_model_path, max_seq_len, max_entity_len, max_attrname_len, max_attrvalue_len):
        
        self.device = torch.device(gpu)

        print('Load model...')
        self.model = ExtractorModel(model_path=pretrain_model_path,
                                    device=self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(save_model_path, "best_model.pt"),
                       map_location=torch.device('cpu')))
        print('Model created!')
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
        self.max_seq_len = max_seq_len
        self.max_entity_len = max_entity_len
        self.max_attrname_len = max_attrname_len
        self.max_attrvalue_len = max_attrvalue_len

    def text_smiliary(self, text1, text2, text3, text4, bacth_size = 64):
        '''
        text1,text2,text3: list(str)
        '''

        sep = self.tokenizer.sep_token

        text1 = [truncate_str(sep.join(context), self.max_seq_len) for context in text1]
        text2 = [truncate_str(t, self.max_entity_len) for t in text2]
        text3 = [truncate_str(t, self.max_attrname_len) for t in text3]
        # text4 = [truncate_str(t, self.max_attrvalue_len) for t in text4]

        text5 = [t2+self.tokenizer.sep_token+t3 for t2,t3 in zip(text2, text3)]

        
        max_length = self.max_seq_len+self.max_entity_len+self.max_attrname_len+4
        inputs = self.tokenizer(
            text1, text5, 
            max_length=max_length,
            truncation=True, padding='max_length', return_tensors='pt'
            )
        n_sample = len(text2)
        n_bacth = (n_sample-1)//bacth_size + 1

        all_logits= []
        for i in range(n_bacth):
            token_ids = inputs["input_ids"][i*bacth_size:(i+1)*bacth_size,:].to(self.device)
            attention_mask = inputs["attention_mask"][i*bacth_size:(i+1)*bacth_size,:].to(self.device)
            token_type_ids = inputs["token_type_ids"][i*bacth_size:(i+1)*bacth_size,:].to(self.device)
            logit = self.model( token_ids=token_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids
                                )

            all_logits.append(logit.detach().cpu().numpy())
        all_logits = np.concatenate(all_logits,)
        return all_logits
