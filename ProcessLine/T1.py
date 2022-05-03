'''
Author: CT
Date: 2022-04-04 19:16:21
LastEditTime: 2022-04-26 18:45:45
'''
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
##################################################################
# Setup
##################################################################
# backbone
backbone_name = "bert-base-uncased"
token_max_length = 32
# encoder
hidden_dim = 768
# decoder
label_nums = 4

##################################################################
# T1
##################################################################
class T1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BertModel.from_pretrained(backbone_name)
        self.decoder = nn.Linear(hidden_dim, label_nums)
        
        self.text_tokenizer = BertTokenizer.from_pretrained(backbone_name)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text):
        # backbone
        text_token = self.text_tokenizer(text, 
                                return_tensors="pt",
                                padding="max_length",
                                max_length=token_max_length,
                                truncation=True).input_ids.to(self.backbone.device)
        hidden_feature = self.backbone(text_token).pooler_output
        # decode
        outputs = self.decoder(hidden_feature)
        # logits
        logits = torch.sigmoid(outputs)
        return logits

    def loss(self, logits, label):
        loss = self.criterion(logits, label)
        return loss

if __name__ == "__main__":
    print(T1())
        