from utils import *
import evaluation
import data_loader
from tqdm import tqdm
import time
import sys
import copy
import argparse
import os
from heapq import heappush, heappop

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)

parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--negNum_train', type=int, default=4)
parser.add_argument('--negNum_test', type=int, default=1000)
parser.add_argument('--train_neg_samples_pool_size', type=int, default=20)
parser.add_argument('--latent_dim', type=int, default=64)

parser.add_argument('--train_device', type=str, default='cuda')
parser.add_argument('--test_device', type=str, default='cpu')
parser.add_argument('--dataloader_workers', type=int, default=4)
parser.add_argument('--val_per_train', type=int, default=5)

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, required=True, default=1e-1)
parser.add_argument('--ref_point_lr', type=float, default=1e-1)
parser.add_argument('--globalBias_lr', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--lambda_userBias_greek', type=float, default=0)
parser.add_argument('--lambda_globalBias', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.1)

parser.add_argument('--show_pbar', type=str2bool, default=True)
parser.add_argument('--save_model', type=str2bool, default=True)
parser.add_argument('--print_epoch_loss', type=str2bool, default=True)

# parser.add_argument()

args = parser.parse_args()
# config = vars(args)

train, val, test = data_loader.read_data(args.dataset)
item_price = data_loader.get_price(args.dataset)

if args.dataset == 'Baby':
	userNum, itemNum = 23894, 39767
else:
	userNum, itemNum = data_loader.get_datasize(args.dataset)

frequency = data_loader.get_distribution(args.dataset)
distribution = data_loader.approx_Gaussian(frequency)

trainset = data_loader.TransactionData(train, userNum, itemNum, distribution, args.train_neg_samples_pool_size)
trainset.set_negN(args.negNum_train)
trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.dataloader_workers)

valset = data_loader.UserTransactionData(val, userNum, itemNum, item_price, trainset.userHist)
valset.set_negN(args.negNum_test)
valLoader = DataLoader(valset, batch_size=1, shuffle=False)

testset = data_loader.UserTransactionData(test, userNum, itemNum, item_price, trainset.userHist)
testset.set_negN(args.negNum_test)
testLoader = DataLoader(testset, batch_size=1, shuffle=False)

if args.model == 'Prelec':
	model = Prelec(userLen=userNum, itemLen=itemNum, distribution=distribution, item_price=item_price, params=vars(args))
elif args.model == 'Prelec+':
	model = Prelec_plus(userLen=userNum, itemLen=itemNum, distribution=distribution, item_price=item_price, params=vars(args))
elif args.model == 'TF':
	model = TF(userLen=userNum, itemLen=itemNum, distribution=distribution, item_price=item_price, params=vars(args))
elif args.model == 'TF+':
	model = TF_plus(userLen=userNum, itemLen=itemNum, distribution=distribution, item_price=item_price, params=vars(args))

if args.optimizer=='sgd':
	optimizer = optim.SGD([
		{'params': model.globalBias_p.parameters(), 'weight_decay': args.lambda_globalBias, 'lr': args.globalBias_lr},
		{'params': model.globalBias_n.parameters(), 'weight_decay': args.lambda_globalBias, 'lr': args.globalBias_lr},
		{'params': model.globalBias_g.parameters(), 'weight_decay': args.lambda_globalBias, 'lr': args.globalBias_lr},
		{'params': model.globalBias_d.parameters(), 'weight_decay': args.lambda_globalBias, 'lr': args.globalBias_lr},
		{'params': model.globalBias_t.parameters(), 'weight_decay': args.lambda_globalBias, 'lr': args.globalBias_lr},
		{'params': model.reference_point.parameters(), 'weight_decay': 0, 'lr': args.ref_point_lr},

		{'params': model.userBias_p.parameters()},
		{'params': model.itemBias_p.parameters()},
		{'params': model.userEmbed_p.parameters()},
		{'params': model.itemEmbed_p.parameters()},

		{'params': model.userBias_n.parameters()},
		{'params': model.itemBias_n.parameters()},
		{'params': model.userEmbed_n.parameters()},
		{'params': model.itemEmbed_n.parameters()},

		{'params': model.userBias_g.parameters(), 'weight_decay': args.lambda_userBias_greek},
		{'params': model.userBias_d.parameters(), 'weight_decay': args.lambda_userBias_greek},
		{'params': model.userBias_t.parameters(), 'weight_decay': args.lambda_userBias_greek},
	],
	lr = args.lr,
	weight_decay = args.weight_decay,
	momentum = args.momentum
	)

epoch=0
print('start training......')
while epoch < args.epoch-1:
	model.device=args.train_device
	model.to(model.device)
	epoch += 1
	print('Epoch ', str(epoch), ' training...')
	L = len(trainLoader.dataset)
	if args.show_pbar:
		pbar = tqdm(total = L)
	for i, batchData in enumerate(trainLoader):
		optimizer.zero_grad()
		users = torch.LongTensor(batchData['user']).to(model.device)
		items = torch.LongTensor(batchData['item']).to(model.device)
		negItems = torch.LongTensor(batchData['negItem']).reshape(-1).to(model.device)

		batch_loss = model.loss(users, items, negItems)
		batch_loss.backward()

		optimizer.step()

		optimizer.zero_grad()
		if i == 0:
			total_loss = batch_loss.clone()
		else:
			total_loss += batch_loss.clone()
		if args.show_pbar:
			pbar.update(users.shape[0])
	if args.show_pbar:
		pbar.close()

	folder='model'
	if args.save_model:
		if not os.path.isdir(folder):
			os.mkdir(folder)
		torch.save(model, folder+'/'+args.model+'_'+args.dataset+'.pt')

	if epoch % args.val_per_train == 0:
		print('start val......')
		model.eval()
		model.device = args.test_device
		model.to(model.device)
		L = len(valLoader.dataset)
		if args.show_pbar:
			pbar = tqdm(total=L)
		with torch.no_grad():
			scoreDict = dict()
			for i, batchData in enumerate(testLoader):
				user = torch.LongTensor(batchData['user']).to(model.device)
				posItems = torch.LongTensor(batchData['posItem']).to(model.device)
				negItems = torch.LongTensor(batchData['negItem']).to(model.device)

				items = torch.cat((posItems, negItems), 1).view(-1)
				users = user.expand(items.shape[0])

				score = model.forward(users, items)
				scoreHeap = list()
				for j in range(score.shape[0]):
					gt = False
					if j < posItems.shape[1]:
						gt = True
					
					heappush(scoreHeap, (1-score[j].cpu().numpy(), (0+items[j].cpu().numpy(), gt)))
				scores = list()
				candidate = len(scoreHeap)
				for k in range(candidate):
					scores.append(heappop(scoreHeap))
				if args.show_pbar:
					pbar.update(1)
				scoreDict[user[0]] = (scores, posItems.shape[1])
		if args.show_pbar:
			pbar.close()
		valResult = evaluation.ranking_performance(scoreDict, 100)

print('starting test...')
model.device = args.test_device
model.to(model.device)
model.eval()
L = len(testLoader.dataset)
pbar = tqdm(total=L)
with torch.no_grad():
	scoreDict = dict()
	for i, batchData in enumerate(testLoader):
		user = torch.LongTensor(batchData['user']).to(model.device)
		posItems = torch.LongTensor(batchData['posItem']).to(model.device)
		negItems = torch.LongTensor(batchData['negItem']).to(model.device)

		items = torch.cat((posItems, negItems), 1).view(-1)
		users = user.expand(items.shape[0])

		score = model.forward(users, items)
		scoreHeap = list()
		for j in range(score.shape[0]):
			gt = False
			if j < posItems.shape[1]:
				gt = True
			
			heappush(scoreHeap, (1-score[j].cpu().numpy(), (0+items[j].cpu().numpy(), gt)))
		scores = list()
		candidate = len(scoreHeap)
		for k in range(candidate):
			scores.append(heappop(scoreHeap))
		pbar.update(1)
		scoreDict[user[0]] = (scores, posItems.shape[1])
pbar.close()
testResult = evaluation.ranking_performance(scoreDict, 100)
