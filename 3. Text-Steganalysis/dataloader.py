# -*- coding: utf-8 -*-
# Date       ：2022/9/5
# Author     ：Chen Xuekai
# Description：加载yzl_dataset/two_tree/movie先做实验

import pandas as pd
from tqdm import tqdm
import json


def build_dataset(args):
    sample_set = []
    dataset = []
    word2id = {'[PAD]': 0, '[CLS]': 1, '.': 2}
    data = pd.read_excel(args.dataset)
    # 构建词表
    print("building vocabulary...")
    for index, row in tqdm(data.iterrows()):
        sentence, label = row
        sentence = sentence.strip()
        if int(label) == 0:  # cover
            label = int(label)
        elif int(label) == 1:  # stego
            label = int(label)
        word_list = sentence.split(' ')
        sample_set.append((word_list, label))
        for word in word_list:
            if word not in word2id.keys():
                word2id[word] = len(word2id)
    # word_list to id_list
    print("converting text to id_list...")
    for sample in sample_set:
        tokens = ['[CLS]'] + sample[0] + ['.']
        id_list = []
        for token in tokens:
            id_list.append(word2id[token])
        dataset.append((id_list, sample[1]))

    return dataset, word2id  # 句子长度不等，需要逐个batch的padding


# Testing coding...
if __name__ == '__main__':
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser(description='data')
    args = parser.parse_args()
    args.dataset = '../../data/hc1_cover-stego.xlsx'

    data_id_list, word2id = build_dataset(args)
    with open('preprocess/data_id_list.json', 'w', encoding='utf-8') as f:
        json.dump(data_id_list, f)
    with open('preprocess/word2id.json', 'w', encoding='utf-8') as f:
        json.dump(word2id, f)
    print("sample number: ")
    print("train: ", len(data_id_list))
