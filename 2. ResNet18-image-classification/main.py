# -*- coding: utf-8 -*-
# Date        : 2022/4/23
# Author      : Chen Xuekai
# Description : train/valid/test ResNet18 for kaggle cat-vs-dog classification

import os
import sys
import torch
from tqdm import tqdm
from torch import optim
from model import ResNet18
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dataset import MyDataset, dataset_split, dataloader
from sklearn.metrics import classification_report,accuracy_score, recall_score, f1_score, precision_score

# 参数设置
batch_size = 64
epoch_num = 30
lr = 0.001
n_class = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def accuracy(predict, target):
    predict = predict.topk(1).indices.flatten()
    acc = predict.eq(target).sum().item() / len(predict)
    return acc


def evaluate(model, mode_loader, report=False):
    model.eval()
    y_test = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(mode_loader):
            batch_inputs, batch_labels = batch[0].to(device), batch[1].to(device)
            y_test += batch_labels.tolist()
            batch_output = model(batch_inputs)
            batch_output = batch_output.topk(1).indices.flatten().tolist()
            y_pred += batch_output

    print('Accuracy  : %.4f%%' % (100 * accuracy_score(y_test, y_pred)))
    print('Recall    : %.4f%%' % (100 * recall_score(y_test, y_pred, average='weighted')))
    print('Precision : %.4f%%' % (100 * precision_score(y_test, y_pred, average='weighted')))
    print('F1-score  : %.4f%%' % (100 * f1_score(y_test, y_pred, average='weighted')))
    if report:
        print(classification_report(y_test, y_pred))
    return 100 * accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    # 加载数据
    all_data = MyDataset("data")
    train_set, valid_set, test_set = dataset_split(all_data, train_rate=0.7, valid_rate=0.1)
    train_loader = dataloader(train_set, batch_size)
    valid_loader = dataloader(valid_set, batch_size)
    test_loader = dataloader(test_set, batch_size)

    # 初始化模型，定义优化器
    model = ResNet18(n_class)
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)

    # 判断是否加载已有模型，若有，则直接evaluate
    train_or_not = input("输入y/n选择是否重新训练模型：")
    if train_or_not == 'n':
        while True:
            model_path = input("输入并加载已有模型路径：")
            if os.path.isfile(model_path):
                model_param = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(model_param["model_state_dict"])
                optimizer.load_state_dict(model_param["optimizer_state_dict"])
                print("已成功加载模型，正在开始测试...")
                model = model.to(device)
                acc = evaluate(model, test_loader, report=True)
                sys.exit()
            else:
                exit_or_not = input("文件路径不存在，输入exit结束程序，或按任意键并重新输入模型路径：")
                if exit_or_not == "exit":
                    sys.exit()
    else:
        # train
        print("training...")
        model = model.to(device)
        writer = SummaryWriter()  # tensorboard绘制曲线
        max_acc = 0  # 记录最高验证集
        no_improve_epoch = 0  # 记录连续几轮性能没有提升
        for epoch in range(epoch_num):
            model.train()
            for idx, batch in enumerate(train_loader):
                inputs, labels = batch[0].to(device), batch[1].to(device)
                output = model(inputs)
                loss = F.cross_entropy(output, labels)
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 计算并打印准确率
                acc = accuracy(output, labels)
                if idx % 10 == 0:
                    print(
                        "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
                        % (epoch, idx, len(train_loader), loss.item(), acc)
                    )

                # tensorboard绘图
                step = epoch * len(train_loader) + idx
                writer.add_scalar("loss/training", loss.item(), step)
                writer.add_scalar("accuracy/training", acc, step)
            # 验证
            valid_acc = evaluate(model, valid_loader)
            writer.add_scalar("accuracy/evaluate", valid_acc, epoch)
            # 若验证效果比前面好，则保存模型
            if valid_acc > max_acc:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    "epoch_%d_%.4f.pth" % (epoch, valid_acc)
                )
                max_acc = valid_acc
                no_improve_epoch = 0
            else:
                no_improve_epoch += 1
            # 若连续5轮没有提升，则停止
            if no_improve_epoch >= 5:
                break

