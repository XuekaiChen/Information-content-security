# -*- coding: utf-8 -*-
# Date       ：2022/9/5
# Author     ：Chen Xuekai
# Description：加载yzl_dataset/two_tree/movie先做实验

import torch
import random
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

PAD, CLS = '[PAD]', '[CLS]'


def build_dataset(args):
    def load_dataset(path, pad_size=300):
        '''
        path: 'hc1_cover-stego.xlsx'
        '''
        contents = []
        data = pd.read_excel(path)
        for index, row in tqdm(data.iterrows()):
            sentence, label = row
            sentence = sentence.strip()
            if int(label) == 0:  # cover
                label = int(label)
            elif int(label) == 1:  # stego
                label = int(label)
            else:  # target
                label = int(label)
            token = args.tokenizer.tokenize(sentence)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = args.tokenizer.convert_tokens_to_ids(token)
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + \
                           [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, label, seq_len, mask))
        # random.shuffle(contents)
        return contents

    dataset = load_dataset(args.dataset)
    # train:valid:test = 8:1:1
    train = dataset[:int(len(dataset)*0.8)]
    valid = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
    test = dataset[int(len(dataset)*0.9):]
    return train, valid, test


class DatasetIterater(object):
    def __init__(self, batches, args):
        self.batch_size = args.batch_size
        self.batches = batches
        self.n_batches = len(batches) // args.batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = args.device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.batches[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, args):
    iters = DatasetIterater(dataset, args)
    return iters


# Testing coding...
if __name__ == '__main__':
    import argparse
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = argparse.ArgumentParser(description='data')
    args = parser.parse_args()
    args.dataset = 'hc1_cover-stego.xlsx'

    args.tokenizer = BertTokenizer.from_pretrained('/data/chenxuekai/bert-base-uncased/')
    args.batch_size = 32
    args.device = 'cuda'

    train_data, valid_data, test_data = build_dataset(args)

    train_iter = build_iterator(train_data, args)
    valid_iter = build_iterator(valid_data, args)
    test_iter = build_iterator(test_data, args)
    print("sample number: ")
    print("train: ", len(train_data))
    print("valid: ", len(valid_data))
    print("test: ", len(test_data))
    print("iterator: ")
    print(len(train_iter))
    print(len(valid_iter))
    print(len(test_iter))

