# -*- coding: utf-8 -*-
# Date        : 2022/3/29
# Author      : Chen Xuekai
# Description : Organize the sentences into regular forms of fixed length

import jieba
import json

class2id = {
    'news_story': 0,
    'news_culture': 1,
    'news_entertainment': 2,
    'news_sports': 3,
    'news_finance': 4,
    'news_house': 5,
    'news_car': 6,
    'news_edu': 7,
    'news_tech': 8,
    'news_military': 9,
    'news_travel': 10,
    'news_world': 11,
    'stock': 12,
    'news_agriculture': 13,
    'news_game': 14
}
id2class = {v: k for k, v in class2id.items()}


# id_list为不定长序列，list类型
def data_2_id_list():
    news_id, class_name, key_words = [], [], []
    target = []
    content_list = []
    input_sentence = []
    # 读取数据
    with open("toutiao_cat_data.txt", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line_ele = line.strip().split('_!_')
            news_id.append(line_ele[0])
            target.append(class2id[line_ele[2]])  # int_list: [data_len,]
            tokens = [i for i in jieba.cut(line_ele[3])]
            content_list.append(tokens)
            if line_ele[4] == "":
                key_words.append([])
            else:
                key_words.append(line_ele[4].split(','))

    assert len(content_list) == len(target)  # 确保sentence和tag一对一
    print("Total samples number: ", len(content_list))
    # 构建词表
    word2id = {'PAD': 0}
    for sentence in content_list:
        for token in sentence:
            if token not in word2id.keys():
                word2id[token] = len(word2id)
    id2word = {v: k for k, v in word2id.items()}

    # token转id
    sentence = []
    for line in content_list:
        for token in line:
            sentence.append(word2id[token])
        input_sentence.append(sentence)
        sentence = []

    return list(zip(input_sentence, target)), word2id, id2word  # [([1,4,2],0),([2,3,4],3)]


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


if __name__ == '__main__':
    news_data, word2id, id2word = data_2_id_list()
    # 暂存
    with open('preprocess/news_data_id_list.json', 'w', encoding='utf-8') as f:
        json.dump(news_data, f)
    with open('preprocess/word2id.json', 'w', encoding='utf-8') as f:
        json.dump(word2id, f)
    with open('preprocess/id2word.json', 'w', encoding='utf-8') as f:
        json.dump(id2word, f)


