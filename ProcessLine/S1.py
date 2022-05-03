'''
Author: CT
Date: 2022-04-04 11:09:32
LastEditTime: 2022-04-12 13:35:14
'''
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
##################################################################
# Setup
##################################################################
# backbone
backbone_name = "facebook/wav2vec2-base"
hidden_dim = 768
# decoder
label_nums = 4

##################################################################
# S1
##################################################################
class S1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        self.processor = Wav2Vec2Processor.from_pretrained(backbone_name)
        self.decoder = nn.Linear(hidden_dim, label_nums)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, speech):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(self.backbone.device)
        inputs = torch.squeeze(inputs, dim=0)
        # backbone
        hidden_feature = self.backbone(inputs).last_hidden_state
        # decode
        outputs = self.decoder(torch.mean(hidden_feature, dim=1))
        # logits
        logits = torch.sigmoid(outputs)

        return logits

    def loss(self, logits, label):
        loss = self.criterion(logits, label)
        return loss

if __name__ == "__main__":
    print(S1())
        