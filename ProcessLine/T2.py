'''
Author: CT
Date: 2022-04-04 10:50:42
LastEditTime: 2022-04-26 18:45:35
'''
import re
import torch
import torch.nn as nn
from transformers import XLNetTokenizer, XLNetModel
##################################################################
# Setup
##################################################################
# backbone
backbone_name = "xlnet-base-cased"
token_max_length = 32
hidden_dim = 768
# decoder
label_nums = 4

##################################################################
# T2
##################################################################
class T2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = XLNetModel.from_pretrained(backbone_name)
        self.decoder = nn.Linear(hidden_dim, label_nums)
        
        self.text_tokenizer = XLNetTokenizer.from_pretrained(backbone_name)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text):
        # backbone
        text_token = self.text_tokenizer(text, 
                                return_tensors="pt",
                                padding="max_length",
                                max_length=token_max_length,
                                truncation=True).input_ids.to(self.backbone.device)
        hidden_feature = self.backbone(text_token).last_hidden_state[:,-1,:]
        # decode
        outputs = self.decoder(hidden_feature)
        # logits
        logits = torch.sigmoid(outputs)
        return logits

    def loss(self, logits, label):
        loss = self.criterion(logits, label)
        return loss

if __name__ == "__main__":
    print(T2())
        