# -*- coding: utf-8 -*-
# Date        : 2022/3/29
# Author      : Chen Xuekai
# Description : train/valid/test RNN model for Toutiao dataset classification

import torch
import json
import os
import sys
import random
import math
from tqdm import tqdm
from operator import itemgetter
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import MyRNN
from data_process import class2id, data_2_id_list, sort_by_lengths, pad_sentence
from sklearn.metrics import classification_report,accuracy_score, recall_score, f1_score, precision_score

# 参数设置
batch_size = 64
lr = 0.001
num_layers = 2
epoch_num = 10
emb_size = 300
hidden_size = 128
n_class = len(class2id)


# 对每个batch做padding
def regularize_batch(mode_set, idx, B):
    batch_sents = list(map(itemgetter(0), mode_set[idx:idx + B]))
    batch_tag = list(map(itemgetter(1), mode_set[idx:idx + B]))
    # padding
    input_lists, target_list = sort_by_lengths(batch_sents, batch_tag)
    input_tensors, lengths, target_tensor = pad_sentence(input_lists, target_list)
    return input_tensors, lengths, target_tensor


# 对所有batch做padding
def regularize_data(mode_set):  # [sample_num*[[input_list],target]]
    mode_inputs = []
    mode_lengths = []
    mode_targets = []
    for idx in range(0, len(mode_set), batch_size):
        input_list, length, target_list = regularize_batch(mode_set, idx, batch_size)
        list(map(lambda x: mode_inputs.append(x), input_list))
        list(map(lambda x: mode_lengths.append(x), length))
        list(map(lambda x: mode_targets.append(x), target_list))
    return (
        mode_inputs,
        mode_lengths,
        mode_targets
    )  # batch内定长，总数据不定长


# 计算predict与target准确率 tensor
def accuracy(predict, target):
    predict = predict.topk(1).indices.flatten()
    acc = predict.eq(target).sum().item() / len(predict)
    return predict, acc


# 对test或valid计算准确率
def evaluate(rnn_model, mode_set, report=False):
    rnn_model.eval()
    epoch_acc = 0
    y_test = []
    y_pred = []
    with torch.no_grad():
        for idx in tqdm(range(0, len(mode_set[0]), batch_size)):
            # 依次按batch取数据，并list转tensor
            target_lists_ = mode_set[2][idx: idx + batch_size]
            y_test += target_lists_
            input_lists_ = mode_set[0][idx: idx + batch_size]
            input_tensors_ = torch.LongTensor(input_lists_)
            output_ = model(input_tensors_)  # [Batch_size, n_class]
            output_ = output_.topk(1).indices.flatten().tolist()
            y_pred += output_

    print('Accuracy  : %.4f%%' % (100 * accuracy_score(y_test, y_pred)))
    print('Recall    : %.4f%%' % (100 * recall_score(y_test, y_pred, average='weighted')))
    print('Precision : %.4f%%' % (100 * precision_score(y_test, y_pred, average='weighted')))
    print('F1-score  : %.4f%%' % (100 * f1_score(y_test, y_pred, average='weighted')))
    if report:
        print(classification_report(y_test, y_pred))
    return 100 * accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    # 加载数据
    news_data_path = "preprocess/news_data_id_list.json"
    word2id_path = "preprocess/word2id.json"
    id2word_path = "preprocess/id2word.json"
    if os.path.isfile(news_data_path) and \
            os.path.isfile(word2id_path) and \
            os.path.isfile(id2word_path):
        with open(news_data_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)
        with open(word2id_path, 'r', encoding='utf-8') as f:
            word2id = json.load(f)
        with open(id2word_path, 'r', encoding='utf-8') as f:
            id2word = json.load(f)
    else:
        news_data, word2id, id2word = data_2_id_list()  # id2word在分析badcase时还有用

    # 分割train/valid/test，因为顺序样本类别分布问题，需要随机打乱
    random.shuffle(news_data)
    train_sample_num = int(0.7*len(news_data))
    valid_sample_num = int(0.1*len(news_data))
    unnorm_train_set = news_data[:train_sample_num]  # [sample_num*([input_list],target)]
    unnorm_valid_set = news_data[train_sample_num: train_sample_num + valid_sample_num]
    unnorm_test_set = news_data[train_sample_num + valid_sample_num:]
    # # 对train/valid/test集内所有batch做padding
    train_set = regularize_data(unnorm_train_set)  # (sam_num*[input_list],sam_num*[length],sam_num*[target])
    valid_set = regularize_data(unnorm_valid_set)
    test_set = regularize_data(unnorm_test_set)
    print("成功加载并预处理数据集...")

    # 获取与数据集有关的其他参数
    vocab_size = len(word2id)
    batch_num = int(train_sample_num / batch_size)
    print("batch number: ", batch_num)

    # 加载模型，定义优化器
    model = MyRNN(vocab_size=vocab_size,
                  emb_size=emb_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  out_size=n_class)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("成功初始化模型...")

    # 判断是否加载已有模型，若有，则可直接evaluate
    train_or_not = input("输入y/n选择是否重新训练模型：")
    if train_or_not == 'n':
        while True:
            model_path = input("输入并加载已有模型路径：")
            if os.path.isfile(model_path):
                model_param = torch.load(model_path)
                model.load_state_dict(model_param["model_state_dict"])
                optimizer.load_state_dict(model_param["optimizer_state_dict"])
                print("已成功加载模型，正在开始测试...")
                test_acc = evaluate(model, test_set, report=True)
                sys.exit()
            else:
                exit_or_not = input("文件路径不存在，输入exit结束程序，或按任意键并重新输入模型路径：")
                if exit_or_not == "exit":
                    sys.exit()
    else:
        # train
        print("training...")
        writer = SummaryWriter()  # tensorboard绘制曲线
        max_acc = 0   # 记录最高验证集
        no_improve_epoch = 0  # 记录连续几轮性能没有提升
        for epoch in range(epoch_num):
            model.train()
            # 由于输入序列不定长，因此不能封装成tensor之后用dataloader加载，只能手动按batch切割
            for idx in range(0, len(train_set[0]), batch_size):
                # 依次按batch取数据，并list转tensor
                input_lists = train_set[0][idx: idx + batch_size]
                length_lists = train_set[1][idx: idx + batch_size]
                target_lists = train_set[2][idx: idx + batch_size]
                input_tensors = torch.LongTensor(input_lists)
                length_tensors = torch.LongTensor(length_lists)
                target_tensors = torch.LongTensor(target_lists)
                # forward
                output = model(input_tensors)  # [Batch_size, n_class]
                # 计算损失
                loss = F.cross_entropy(output, target_tensors)
                # 损失回传
                optimizer.zero_grad()
                loss.backward()
                # 参数更新
                optimizer.step()
                # 计算并打印准确率
                predict, acc = accuracy(output, target_tensors)
                if int(idx / batch_size) % 100 == 0:
                    print(
                        "Training: Epoch=%d, Batch=%d/%d, Loss=%.4f, Accuracy=%.4f"
                        % (epoch, int(idx / batch_size), batch_num, loss.item(), acc)
                    )
                # 绘制loss/accuracy曲线
                step = epoch * train_sample_num + idx
                writer.add_scalar("loss/training", loss.item(), step)
                writer.add_scalar("accuracy/training", acc, step)
            # 验证
            valid_acc = evaluate(model, valid_set)   # 百分制
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
            # 若连续3轮没有提升，则停止
            if no_improve_epoch >= 3:
                break
