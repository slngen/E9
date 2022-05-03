'''
Author: CT
Date: 2022-04-07 15:38:35
LastEditTime: 2022-04-13 19:13:46
'''
import re
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.nn.functional import log_softmax
import torch.nn.functional as F
##################################################################
# Setup
##################################################################
# backbone
backbone_name = "facebook/wav2vec2-base"
hidden_dim = 768
# cls decoder
label_nums = 4
# ctc decoder
vocab_nums = 32
# multitask
alpha = 0.1

##################################################################
# M0
##################################################################
class M0(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(backbone_name)
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        self.ctc_decoder = nn.Linear(hidden_dim, vocab_nums)
        self.cls_decoder = nn.Linear(hidden_dim, label_nums)

        self.criterion_CE = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, speech):
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(self.backbone.device)
        inputs = torch.squeeze(inputs, dim=0)
        # backbone
        hidden_feature = self.backbone(inputs).last_hidden_state
        # ctc
        outputs_ctc = self.ctc_decoder(hidden_feature)
        logits_ctc = self.log_softmax(outputs_ctc)
        # cls
        hidden_feature = torch.mean(hidden_feature, dim=1)
        outputs_cls = self.cls_decoder(hidden_feature)
        logits_cls = torch.sigmoid(outputs_cls)
        # logits
        logits = {"ctc": logits_ctc, "cls": logits_cls}
        return logits

    def loss(self, logits, label):
        with self.processor.as_target_processor():
            label_ctc = self.processor.tokenizer.clean_up_tokenization(label["ctc"][0])
            label_ctc = self.processor(label_ctc, return_tensors="pt").input_ids
            label_ctc = torch.squeeze(label_ctc)
            label_ctc[-1] = 4
            loss_ctc = self._ctc_loss(logits["ctc"], label_ctc)
        loss_cls = self._cls_loss(logits["cls"], label["cls"])
        loss = loss_cls + self.alpha * loss_ctc
        return {"total": loss, "cls": loss_cls, "ctc": loss_ctc}

    def _ctc_loss(self, logits, label):
        loss = F.ctc_loss(logits.transpose(0, 1), label,(logits.shape[1],), (label.shape[0],), reduction="sum")
        return loss

    def _cls_loss(self, logits, label):
        loss = self.criterion_CE(logits, label)
        return loss


if __name__ == "__main__":
    print(M0())
