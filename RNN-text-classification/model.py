# -*- coding: utf-8 -*-
# Date        : 2022/3/28
# Author      : Chen Xuekai
# Description : Design RNN model and verify its availability

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_process import data_2_id_list, sort_by_lengths, pad_sentence


class MyRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(MyRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(input_size=emb_size,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=n_layers,
                          bidirectional=True)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor):
        embed = self.embedding(sents_tensor)  # [B, L, emb_size]

        # 输出维度：rnn_out:[B, L, hidden_size*2]
        # rnn_out, last_hidden = self.rnn(embed)  # 使用rnn
        rnn_out, (h, c) = self.lstm(embed)  # 使用lstm
        rnn_out = rnn_out[:, -1, :]  # [B,L,2*hidden_size]-->[B,2*hidden_size]
        scores = self.fc(rnn_out)  # [B, out_size]
        return scores


if __name__ == "__main__":
    # 模拟测试
    vocab_size = 37682
    emb_size = 100
    hidden_size = 64
    n_layers = 3
    out_size = 6
    batch_sents = [[1,2,3,4,5],[6,7,4,2,1],[4,2,4],[4,3]]
    batch_tag = [3,2,1,0]
    input_lists, target_list = sort_by_lengths(batch_sents, batch_tag)
    input_lists, length_list, target_list = pad_sentence(input_lists, target_list)
    input_tensors = torch.LongTensor(input_lists)
    lengths = torch.LongTensor(length_list)
    target_tensor = torch.LongTensor(target_list)
    model = MyRNN(vocab_size=vocab_size,
                  emb_size=emb_size,
                  hidden_size=hidden_size,
                  n_layers=n_layers,
                  out_size=out_size)
    print(model)
    model.train()
    score = model(input_tensors)
    print(score.shape)
    print(score)
    loss = F.cross_entropy(score, target_tensor)
    a = score.topk(1).indices.flatten()
    print("predict: ", a)
    print("target: ", target_tensor)
    acc = a.eq(target_tensor).sum().item() / len(a)
    print(acc)
