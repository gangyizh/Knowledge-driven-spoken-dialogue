import torch
import torch.nn as nn
from transformers import AutoModel,AutoConfig
import torch.nn.functional as F

torch.manual_seed(1)


class ExtractorModel(nn.Module):
    def __init__(self, device, model_path):
        super(ExtractorModel, self).__init__()
        self.device = device

        self.model = AutoModel.from_pretrained(model_path)
        for param in self.model.parameters():
            param.requires_grad = True
        model_config = AutoConfig.from_pretrained(model_path)
        hidden_size = model_config.hidden_size
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, token_ids, attention_mask,
                token_type_ids):
        outputs = self.model(token_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)

        cls_emb = outputs.last_hidden_state[:, 0]
        # print('cls_emb shape:',cls_emb.shape)
        logit = self.out_layer(cls_emb)
        return logit
