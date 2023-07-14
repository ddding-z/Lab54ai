import os
import re
import pandas as pd
import torch
from torch import nn
import argparse
from dataset import MyDataset
from model import MyModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

BATCH_SIZE = 32
learning_rate = 3e-5
EPOCH = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = './dataset/data'
train_label_path = './dataset/train.txt'
file_list = os.listdir(data_path)


def train():
    """
    load data
    """
    print("-----Start Loading Data-----")
    data_type = 'multi'
    train_data = pd.read_csv(train_label_path)
    train_pics = []
    train_texts = []
    for item in train_data['guid']:
        pic_path = os.path.join(data_path, str(item) + ".jpg")
        train_pics.append(pic_path)
        text_path = os.path.join(data_path, str(item) + ".txt")
        # 文本数据是一些推文，所以包含许多@信息和#打头的tag，我认为这是无助于文本特征提取的，所以删去，首先对文本数据做一定的清洗。
        #  ...删除txt中的链接和一些@xxx的无用信息...
        # 但是由于我没有做对比实验，所以其实这个方法的有效性还有待商榷。
        rawtext = open(text_path, 'r', encoding='utf-8', errors='ignore').read()
        rawtext = rawtext.replace("RT ", '').replace('#', '')
        text = re.sub('@\w+\s?', '', rawtext)
        train_texts.append(text)
    train_data['pic'] = train_pics
    train_data['text'] = train_texts
    # 划分验证集
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=12)
    train_data.reset_index(inplace=True)
    valid_data.reset_index(inplace=True)

    train_dataset = MyDataset(train_data, data_type)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataset = MyDataset(valid_data, data_type)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = MyModel()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"-----Start Training-----")
    Best_Acc = 0
    for epoch in range(EPOCH):
        accuracy_sum, loss_sum = 0.0, 0.0
        model.train()
        for images, input_ids, token_type_ids, attention_mask, tag, guid in train_dataloader:
            images = images.to(DEVICE)
            input_ids = input_ids.squeeze().to(DEVICE)
            token_type_ids = token_type_ids.squeeze().to(DEVICE)
            attention_mask = attention_mask.squeeze().to(DEVICE)
            tags = tag.to(DEVICE)

            outputs = model(images, input_ids, token_type_ids, attention_mask)
            loss = criterion(outputs, tags)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            accuracy_sum += torch.sum(preds == tags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss = loss_sum / len(train_dataloader)
        Acc = accuracy_sum.item() / len(train_dataloader.dataset)
        print(f"Epoch {epoch + 1}, Train Loss: {Loss}, Train Acc: {Acc}")
        # break
        accuracy_sum, loss_sum = 0.0, 0.0
        with torch.no_grad():
            for images, input_ids, token_type_ids, attention_mask, tag, guid in valid_dataloader:
                images = images.to(DEVICE)
                input_ids = input_ids.squeeze().to(DEVICE)
                token_type_ids = token_type_ids.squeeze().to(DEVICE)
                attention_mask = attention_mask.squeeze().to(DEVICE)
                tags = tag.to(DEVICE)

                outputs = model(images, input_ids, token_type_ids, attention_mask)
                loss = criterion(outputs, tags)
                loss_sum += loss.item()
                _, preds = torch.max(outputs, 1)
                accuracy_sum += torch.sum(preds == tags)
        Loss = loss_sum / len(valid_dataloader)
        Acc = accuracy_sum.item() / len(valid_dataloader.dataset)
        print(f"Epoch {epoch + 1}, Valid Loss: {Loss}, Valid Acc: {Acc}")

        if Acc > Best_Acc:
            Best_Acc = Acc
            torch.save(model, 'model/best_model.pth')
            print("save sucessfully!")
        print(f"Epoch {epoch + 1} finished")


if __name__ == '__main__':
    train()
