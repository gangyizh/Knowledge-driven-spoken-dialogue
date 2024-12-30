from matplotlib.pyplot import text
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def truncate_str(x, m_len):
    '''
    文本超出范围则取前后部分
    '''
    if len(x)<=m_len:
        return x
    else:
        return x[:(m_len+1)//2] + x[-(m_len//2):]



class DatasetExtractor(Dataset):
    def __init__(self, data, max_seq_len, max_attrname_len, max_attrvalue_len, model_path):
        self.data = data
        self.max_seq_len = max_seq_len
        self.max_attrname_len = max_attrname_len
        self.max_attrvalue_len = max_attrvalue_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_id, text1, text2, text3, label = self.data[index]
        text1 = truncate_str(text1, self.max_seq_len)
        text2 = truncate_str(text2, self.max_attrname_len)
        text3 = truncate_str(text3, self.max_attrvalue_len)

        text4 = text2+self.tokenizer.sep_token+text3

        sample = self.tokenizer(
            text1, text4, 
            max_length=self.max_seq_len+self.max_attrname_len+self.max_attrvalue_len+4,
            truncation=True, padding='max_length', return_tensors='pt')
        text_id = torch.LongTensor([text_id])
        label = torch.FloatTensor([label])
        return text_id,sample,label
