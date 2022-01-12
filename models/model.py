import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Rater(nn.Module):
    def __init__(self):
        super(Rater, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 3)  # [LOW, MED, HIGH]
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.bert(x).last_hidden_state      # (bs, 512, 768)
        x = x[:, 0, :]         # (bs, 768) take only [cls] token
        x = self.layernorm(x)  # (bs, 768)
        x = self.fc1(x)       # (bs, 256)
        x = self.relu(x)      # (bs, 256)
        out = self.fc2(x)     # (bs, 3)

        return out
