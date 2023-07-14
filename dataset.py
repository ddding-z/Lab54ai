import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')


class MyDataset(Dataset):
    def __init__(self, data, data_type):
        self.data = data
        self.data_type = data_type
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 图片缩放到统一大小
            transforms.ToTensor(),  # 将图片转换为tensor
            # transforms.Normalize([0.5], [0.5])
        ])
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')
        self.tag2id = {'negative': 0, 'positive': 1, 'neutral': 2}

    def __getitem__(self, index):
        guid = self.data.loc[index]['guid']
        tag = self.data.iloc[index]['tag']
        tokenized_tag = -1
        if tag == 'negative' or tag == 'positive' or tag == 'neutral':
            tokenized_tag = self.tag2id[tag]
        pic_path = self.data.iloc[index]['pic']
        if self.data_type == 'text':
            tokenized_pic = torch.zeros(size=(3, 224, 224))
            tokenized_text = self.tokenizer(
                self.data.iloc[index]['text'],
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        elif self.data_type == 'pic':
            tokenized_pic = self.transform(Image.open(pic_path))
            tokenized_text = self.tokenizer(
                '',
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        else:
            tokenized_pic = self.transform(Image.open(pic_path))
            tokenized_text = self.tokenizer(
                self.data.iloc[index]['text'],
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        # tokenized_text['input_ids'] = tokenized_text['input_ids'].squeeze()
        # tokenized_text['token_type_ids'] = tokenized_text['token_type_ids'].squeeze()
        # tokenized_text['attention_mask'] = tokenized_text['attention_mask'].squeeze()

        return tokenized_pic, tokenized_text['input_ids'], tokenized_text['token_type_ids'], tokenized_text[
            'attention_mask'], tokenized_tag, guid

    def __len__(self):
        return len(self.data)

