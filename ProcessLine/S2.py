'''
Author: CT
Date: 2022-04-04 19:09:33
LastEditTime: 2022-04-12 13:35:02
'''
import torch
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2Processor
##################################################################
# Setup
##################################################################
# backbone
backbone_name = "microsoft/wavlm-base"
hidden_dim = 768
# decoder
label_nums = 4

##################################################################
# S2
##################################################################
class S2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(backbone_name)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
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
    print(S2())
        