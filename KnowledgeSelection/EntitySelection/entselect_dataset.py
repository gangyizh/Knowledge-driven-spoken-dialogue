from matplotlib.pyplot import text
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer,BertTokenizer


def truncate_str(x, m_len):
    '''
    文本超出范围则取前后部分
    '''
    if len(x)<=m_len:
        return x
    else:
        return x[:(m_len+1)//2] + x[-(m_len//2):]



class DatasetExtractor(Dataset):
    def __init__(self, data, max_seq_len, max_entity_len, max_attrname_len, max_attrvalue_len, model_path):
        self.data = data
        self.max_seq_len = max_seq_len
        self.max_entity_len = max_entity_len
        self.max_attrname_len = max_attrname_len
        self.max_attrvalue_len = max_attrvalue_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_id, text1, text2, text3, text4, ent_label, attr_label = self.data[index]
        ### text1: context, text2: entity
        context = text1
        sep = self.tokenizer.sep_token
        context = sep.join(context)
        text1 = truncate_str(context, self.max_seq_len)
        text2 = truncate_str(text2, self.max_entity_len)
        text3 = truncate_str(text3, self.max_attrname_len)
        text5 = text2+self.tokenizer.sep_token+ text3

        max_length = self.max_seq_len+self.max_entity_len+self.max_attrname_len+4
        sample = self.tokenizer(
            text1, text5, 
            max_length=max_length,
            truncation=True, padding='max_length', return_tensors='pt')
        text_id = torch.LongTensor([text_id])
        ent_label = torch.FloatTensor([ent_label])
        attr_label = torch.FloatTensor([attr_label])
        return text_id, sample, ent_label, attr_label

    # def sequence_tokenizer(self, text):
    #     sample = self.tokenizer.batch_encode_plus(text,
    #                                               max_length=self.max_seq_len,
    #                                               truncation=True,
    #                                               padding='max_length',
    #                                               return_tensors='pt')

    #     return sample
