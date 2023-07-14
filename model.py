import torch
from torchvision.models import resnet50
from transformers import AutoModel
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-small")
        self.resnet = resnet50(pretrained=True)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Linear(256, 3),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            # nn.Linear(256, 3),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward(self, pic, input_ids, token_type_ids, attention_mask):
        # text
        feature1 = self.bert(input_ids, token_type_ids, attention_mask)
        out1 = self.fc1(feature1['pooler_output'])
        # pic
        feature2 = self.resnet(pic)
        out2 = self.fc2(feature2)
        return self.fc3(torch.cat((out1, out2), dim=1))

