# 实验五：多模态情感分析

## Setup

运行实验代码需要安装以下依赖：

- torchvision==0.15.0
- pytorch==2.0.0
- numpy==1.23.5
- scikit-learn==1.2.1
- pandas==1.5.3
- transformer==4.24.0

在终端中使用如下命令即可安装所需依赖：

```
pip install -r requirements.txt
```

## Project Structure

本项目仓库的文件结构如下：

```
│  .gitignore
│  dataset.py # 定义数据集相关
│  eval.py # 消融实验代码
│  model.py # 模型代码
│  preds.txt # 预测结果文件
│  README.md
│  requirements.txt # 依赖项文件
│  test.py # 生成测试代码
│  train.py # 训练代码
│  
├─dataset #数据集
│  │  test_without_label.txt
│  │  train.txt
│  │  
│  └─data #图像和文本数据
│          
└─model #训练后生成，存放训练好的模型
```

## How to use

训练模型，训练得到的最佳模型会保存在model文件下。

```
python train.py
```

生成测试结果。

```
python test.py --data_type multi #其中--data_type可以填写的参数 text,pic,multi
```

消融实验。

```
python eval.py --data_type pic #只使用图片
python eval.py --data_type text #只使用文本
```