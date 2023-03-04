# -*- coding: utf-8 -*-
# Date       ：2023/1/28
# Author     ：Chen Xuekai
# Description：自构建word2id词典并逐batch做padding测试rnn

import os
import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
import BiRNN_CNN
import random
import train
from dataloader import *
from transformers import BertTokenizer
import warnings
from operator import itemgetter


# 对每个batch按长度由大到小排序
def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists


# 对排序后的句子进行padding，batch内定长输出list
def pad_sentence(word_lists, target_list):
    max_seq_len = len(word_lists[0])
    lengths = [len(sentence) for sentence in word_lists]  # batch每句话的实际长度
    for sentence in word_lists:
        padding = [0] * (max_seq_len - len(sentence))
        sentence += padding
    return (
        word_lists,
        lengths,
        target_list
    )


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
    for idx in range(0, len(mode_set), args.batch_size):
        input_list, length, target_list = regularize_batch(mode_set, idx, args.batch_size)
        list(map(lambda x: mode_inputs.append(x), input_list))
        list(map(lambda x: mode_lengths.append(x), length))
        list(map(lambda x: mode_targets.append(x), target_list))
    return (
        mode_inputs,
        mode_lengths,
        mode_targets
    )  # batch内定长，总数据不定长


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='R_BiLTM_C')

# learning
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default:0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default:20]')
parser.add_argument('-log-interval', type=int, default=20, help='how many steps to wait defore logging train status')
parser.add_argument('-early-stop', type=int, default=10, help='iteration numbers to stop without performace boost')
parser.add_argument('-save-dir', type=str, default='hc1', help='where to save the snapshot')

# model
parser.add_argument('-num_layers', type=int, default=2, help='the number of LSTM layers [default:2]')
parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [defualt:256]')
parser.add_argument('-hidden_size', type=int, default=128, help='the number of hidden unit [defualt:100]')
parser.add_argument('-class_num', type=int, default=2, help='the number of class unit [defualt:2]')
parser.add_argument('-kernel-sizes', type=str, default=[3, 5, 7], help='the sizes of kernels of CNN layers')
parser.add_argument('-kernel-num', type=int, default=128, help='the number of each CNN kernels [default:100]')
parser.add_argument('-LSTM_dropout', type=float, default=0.5, help='the probability for LSTM dropout [defualt:0.5]')
parser.add_argument('-CNN_dropout', type=float, default=0.5, help='the probability for CNN dropout [defualt:0.5]')

# data
parser.add_argument('--dataset', type=str, default='../../data/hc1_cover-stego.xlsx', help='the path of data folder')

# device
parser.add_argument('--device', type=str, default='cuda', help='device to use for trianing [default:cuda]')
parser.add_argument('--idx-gpu', type=str, default='1', help='the number of gpu for training [default:2]')  # TODO

# option
parser.add_argument('-test', type=bool, default=False, help='train or test [default:False]')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu

# set seed
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load data
print('\nLoading data...')
# 加载数据
data_id_list, word2id = build_dataset(args)
args.embed_num = len(word2id)
# train:valid:test = 8:1:1
unnorm_train_set = data_id_list[:int(len(data_id_list) * 0.8)]
unnorm_valid_set = data_id_list[int(len(data_id_list) * 0.8):int(len(data_id_list) * 0.9)]
unnorm_test_set = data_id_list[int(len(data_id_list) * 0.9):]
print("sample number: ")
print("train: {}\tvalid: {}\ttest: {}".format(len(unnorm_train_set), len(unnorm_valid_set), len(unnorm_test_set)))
train_set = regularize_data(unnorm_train_set)  # (sam_num*[input_list],sam_num*[length],sam_num*[target])
valid_set = regularize_data(unnorm_valid_set)
test_set = regularize_data(unnorm_test_set)
print('\nfinish Loading data !')


# model
model = BiRNN_CNN.R_BI_C(args)

# initializing model
for name, w in model.named_parameters():
    if 'embed' not in name:
        if 'fc1.weight' in name:
            nn.init.xavier_normal_(w)

        elif 'bias' in name:
            nn.init.constant_(w, 0)


# Caculate the number of parameters of the loaded model
total_params = sum(p.numel() for p in model.parameters())
print('Model_size: ', total_params)

if torch.cuda.is_available():
    torch.device(args.device)
    model = model.cuda()

# Training
args.test = False
print('-----------training-----------')
train.train(train_set, valid_set, model, args, len(valid_set[0]))

# Testing
args.test = True
print('\n-----------testing-----------')
print('Loading test model from {}...'.format(args.save_dir))
models = []
files = sorted(os.listdir(args.save_dir))
for name in files:
    if name.endswith('.pt'):
        models.append(name)
model_steps = sorted([int(m.split('_')[-1].split('.')[0]) for m in models])
ACC, R, P, F1 = 0, 0, 0, 0
for step in model_steps[-5:]:
    best_model = 'best_epochs_{}.pt'.format(step)
    m_path = os.path.join(args.save_dir, best_model)
    print('the {} model is loaded...'.format(m_path))
    model.load_state_dict(torch.load(m_path))
    acc, r, p, f = train.data_eval(test_set, model, args, len(test_set[0]))
    ACC += acc
    R += r
    P += p
    F1 += f

