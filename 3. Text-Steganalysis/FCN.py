import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class TC_base(nn.Module):
    def __init__(self, in_features, class_num, dropout_rate):
        super(TC_base, self).__init__()
        self.in_features = in_features
        self.dropout_prob = dropout_rate
        self.num_labels = class_num  #
        self.dropout = nn.Dropout(self.dropout_prob)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.in_features, self.num_labels)

    def forward(self, features):
        clf_input = self.pool(features.permute(0, 2, 1)).squeeze()
        logits = self.classifier(clf_input)
        return logits


class FCN(nn.Module):
    def __init__(self, args):
        super(FCN, self).__init__()
        self.args = args
        self.vocab_size = args.embed_num
        self.in_feature = args.embed_dim
        self.dropout_prob = args.dropout
        self.num_labels = args.class_num
        self.embedding = nn.Embedding(self.vocab_size, self.in_feature)
        self.classifier = TC_base(self.in_feature, self.num_labels, self.dropout_prob)

    def forward(self, input_ids):
        clf_input = self.embedding(input_ids.long())
        logits = self.classifier(clf_input)
        return logits
