import os
import re
import pandas as pd
import torch
import argparse
from dataset import MyDataset
from model import MyModel
from torch.utils.data import DataLoader

BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = './dataset/data'
train_label_path = './dataset/train.txt'
test_label_path = './dataset/test_without_label.txt'
file_list = os.listdir(data_path)


def pred(data_type):
    # load test data
    print("-----Start Loading Data-----")
    test_data = pd.read_csv(test_label_path)
    test_pics = []
    test_texts = []
    for item in test_data['guid']:
        pic_path = os.path.join(data_path, str(item) + ".jpg")
        test_pics.append(pic_path)
        text_path = os.path.join(data_path, str(item) + ".txt")
        rawtext = open(text_path, 'r', encoding='utf-8', errors='ignore').read()
        rawtext = rawtext.replace("RT ", '').replace('#', '')
        text = re.sub('@\w+\s?', '', rawtext)
        test_texts.append(text)
    test_data['pic'] = test_pics
    test_data['text'] = test_texts

    test_data.reset_index(inplace=True)
    test_dataset = MyDataset(test_data, data_type)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MyModel()
    model.to(DEVICE)
    model = torch.load('./model/best_model.pth', map_location=DEVICE)
    guids = []
    preds = []
    id2tag = {0: "negative", 1: "positive", 2: "neutral"}
    print(f"-----Start Testing-----")
    model.eval()
    with torch.no_grad():
        for images, input_ids, token_type_ids, attention_mask, tag, guid in test_dataloader:
            images = images.to(DEVICE)
            input_ids = input_ids.squeeze().to(DEVICE)
            token_type_ids = token_type_ids.squeeze().to(DEVICE)
            attention_mask = attention_mask.squeeze().to(DEVICE)
            outputs = model(images, input_ids, token_type_ids, attention_mask)
            _, pred = torch.max(outputs, 1)
            pred = pred.cpu().numpy().tolist()
            for i in guid.cpu().numpy().tolist():
                guids.append(i)
            for i in pred:
                preds.append(id2tag[i])
        print("test finished")
    with open('preds.txt', 'w') as f:
        f.write('guid,tag\n')
        for i in range(len(guids)):
            f.write(str(guids[i]) + ',' + str(preds[i]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='multi', type=str)
    args = parser.parse_args()
    pred(args.data_type)
