import os
import sys
import argparse
import datetime
import torch
from transformers import BertModel, BertTokenizer
import MyBert
import train
from dataloader import *
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='MyBert')

# learning
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 32]')
parser.add_argument('-lr', type=float, default=2e-5, help='initial learning rate [default:5e-5]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default:30]')
parser.add_argument('-log-interval', type=int, default=20, help='how many steps to wait defore logging train status')
parser.add_argument('-early-stop', type=int, default=5, help='iteration numbers to stop without performace boost')
parser.add_argument('-save-dir', type=str, default='snapshot-hc1', help='where to save the snapshot')
parser.add_argument('-load-dir', type=str, default=None, help='where to loading the trained model')

# data
parser.add_argument('-dataset', type=str, default='hc1_cover-stego.xlsx', help='the path of data folder')

# device
parser.add_argument('--device', type=str, default='cuda', help='device to use for trianing [default:cuda]')
parser.add_argument('--idx-gpu', type=str, default='0', help='the number of gpu for training [default:2]')  # TODO

# option
parser.add_argument('-test', type=bool, default=False, help='train or test [default:False]')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu
args.tokenizer = BertTokenizer.from_pretrained('/data/chenxuekai/bert-base-uncased')

# set seed
seed = 123
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load data
print('\nLoading data...')
train_data, valid_data, test_data = build_dataset(args)
print("sample number: ")
print("train: {}\tvalid: {}\ttest: {}".format(len(train_data), len(valid_data), len(test_data)))
train_iter = build_iterator(train_data, args)
valid_iter = build_iterator(valid_data, args)
test_iter = build_iterator(test_data, args)

# model
model = MyBert.MyBert(args)
# Caculate the number of parameters of the loaded model
total_params = sum(p.numel() for p in model.parameters())
print('Model_size: ', total_params)

if torch.cuda.is_available():
    torch.device(args.device)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1]).to(args.device)
    model = model.cuda()

# Training
args.test = False
print('-----------training-----------')
train.train(train_iter, valid_iter, model, args, len(valid_data))

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
    acc, r, p, f = train.data_eval(test_iter, model, args, len(test_data))
    ACC += acc
    R += r
    P += p
    F1 += f

