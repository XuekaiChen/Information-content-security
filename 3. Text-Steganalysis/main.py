import os
import sys
import argparse
import datetime
import torch
import torch.nn as nn
from torchtext.vocab import Vectors
import torchtext.data as data

import FCN
import train
import DataLoader


# 加载参数
parser = argparse.ArgumentParser(description='FCN')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 256]')  # default = 20
parser.add_argument('-log-interval', type=int, default=20, help='how many steps to wait before logging train status')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default: 500]')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance boost')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-load-dir', type=str, default=None, help='where to load the trained model')

# 加载数据集
dataset = 'tweets'
payload = '1bpw'
dataset_load = '../../Dataset/' + dataset + '/Tina-Fang/5bpw/'

parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch [default: False]')
parser.add_argument('-train-cover-dir', type=str, default=dataset_load  + 'train_cover.txt', help='the path of train cover data. [default: tweets_cover.txt]')
parser.add_argument('-train-stego-dir', type=str, default=dataset_load  + 'train_stego.txt', help='the path of train stego data. [default: tweets_stego.txt]')
parser.add_argument('-test-cover-dir', type=str, default=dataset_load  + 'test_cover.txt', help='the path of test cover data. [default: test_cover.txt]')
parser.add_argument('-test-stego-dir', type=str, default=dataset_load  + 'test_stego.txt', help='the path of test stego data. [default: test_stego.txt]')

# 模型参数
parser.add_argument('-num-layers', type=int, default=1, help='the number of LSTM layers [default: 3]')
parser.add_argument('-kernel-sizes', type=str, default=[3, 5, 7], help='the sizes of kernels of CNN layers')
parser.add_argument('-kernel-num', type=int, default=128, help='the number of each CNN kernels [default: 100]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')  # 和glove的维度对应
parser.add_argument('-hidden-size', type=int, default=100, help='the number of hidden unit [default: 300]')  # 300
parser.add_argument('-dropout', type=float, default=0.2, help='the probability for LSTM dropout [default: 0.5]')
parser.add_argument('-CNN_dropout', type=float, default=0.5, help='the probability for CNN dropout [default: 0.5]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu [default: False]')
parser.add_argument('-device', type=str, default='cuda', help='device to use for training [default: cuda]')
parser.add_argument('-idx-gpu', type=str, default='0', help='the number of gpu for training [default: 0]')
parser.add_argument('-test', type=bool, default=True, help='train or test [default: False]')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.idx_gpu
parser.add_argument('n_dim', type=int, default=10, help='n-dimensional seq_len feature [default: 10]')


# data_loader
def data_loader(text_field, label_field, args,  **kwargs):
	train_data, valid_data = DataLoader.MyData.split(text_field, label_field, args, 'train')
	text_field.build_vocab(train_data, valid_data)
	label_field.build_vocab(train_data, valid_data)
	train_iter, valid_iter = data.Iterator.splits((train_data, valid_data), batch_sizes=(args.batch_size, len(valid_data)), **kwargs)
	return train_iter, valid_iter


# 加载数据
print('\n Loading data...')
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, valid_iter = data_loader(text_field, label_field, args, device=args.device, sort=False)

# 更新参数
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda


# 模型初始化
model = FCN.FCN(args)

for name, w in model.named_parameters():
	if 'embed' not in name:
		if 'fc1.weight' in name:
			nn.init.xavier_normal_(w)
		elif 'weight' in name and 'conv' in name:
			nn.init.normal_(w, 0.0, 0.1)
		if 'bias' in name:
			nn.init.constant_(w, 0)

if args.load_dir is not None:
	print('\nLoading model from {}...'.format(args.load_dir))
	model.load_state_dict(torch.load(args.load_dir))

if args.cuda:
	torch.device(args.device)
	model.cuda()


total_params = sum(p.numel() for p in model.parameters())
print('Model_size: ', total_params)


# train
if not args.test:
	import time
	start = time.time()
	train.train(train_iter, valid_iter, model, args)
	end = time.time()
	print('time: ', end - start)


# test
if args.test:
	del train_iter, valid_iter
	test_data = DataLoader.MyData.split(text_field, label_field, args, 'test')
	test_iter = data.Iterator.splits([test_data], batch_sizes=[len(test_data)], device=args.device, sort=False)[0]
	print('\n----------testing------------')
	print('Loading test model from {}...'.format(args.save_dir))
	models = []
	files = sorted(os.listdir(args.save_dir))
	for name in files:
		if name.endswith('.pt'):
			models.append(name)
	model_steps = sorted([int(m.split('_')[-1].split('.')[0]) for m in models])

	for step in model_steps[-3:]:
		best_model = 'best_steps_{}.pt'.format(step)
		m_path = os.path.join(args.save_dir, best_model)
		print('the {} model is loaded...'.format(m_path))
		model.load_state_dict(torch.load(m_path))
		train.data_eval(test_iter, model, args)
