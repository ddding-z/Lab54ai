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

BATCH_SIZE = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = './dataset/data'
train_label_path = './dataset/train.txt'
file_list = os.listdir(data_path)


def evaluate(data_type):
    """
    load data
    """
    print("-----Start Loading Data-----")
    train_data = pd.read_csv(train_label_path)
    train_pics = []
    train_texts = []
    for item in train_data['guid']:
        pic_path = os.path.join(data_path, str(item) + ".jpg")
        train_pics.append(pic_path)
        text_path = os.path.join(data_path, str(item) + ".txt")
        rawtext = open(text_path, 'r', encoding='utf-8', errors='ignore').read()
        rawtext = rawtext.replace("RT ", '').replace('#', '')
        text = re.sub('@\w+\s?', '', rawtext)
        train_texts.append(text)
    train_data['pic'] = train_pics
    train_data['text'] = train_texts
    # 划分验证集
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=12)
    valid_data.reset_index(inplace=True)
    valid_dataset = MyDataset(valid_data, data_type)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # print(f"-----load finish {data_type}-----")
    model = MyModel()
    model.to(DEVICE)
    model = torch.load('model/best_model.pth', map_location=DEVICE)
    # model = torch.load('model/best_model.pth', map_location=DEVICE)
    criterion = nn.CrossEntropyLoss()

    print(f"-----Start Evaluating {data_type}-----")
    model.eval()
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
    print(f"Evaluating {data_type}, Valid Loss: {Loss}, Valid Acc: {Acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='pic', type=str)
    args = parser.parse_args()
    evaluate(args.data_type)
