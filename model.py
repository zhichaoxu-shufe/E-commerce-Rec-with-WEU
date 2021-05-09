import pdb
from heapq import heappush, heappop
from utils import ZeroEmbedding
import evaluation
import data_loader
from tqdm import tqdm
import time
import numpy as np
import sys


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TF(nn.Module):
	def __init__(self, userLen, itemLen, distribution, item_price, params):
		super(TF, self).__init__()
		self.userNum = userLen
		self.itemNum = itemLen
		self.params = params

		if 'gpu' in params and params['gpu'] == True:
			self.device = 'cuda'
		else:
			self.device = 'cpu'

		l_size = params['latent_dim']
		self.distribution = torch.FloatTensor(distribution).to(self.device)
		self.item_price = torch.FloatTensor(item_price).to(self.device)

		self.globalBias_g = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_g.weight.data += 0.5
		self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_d.weight.data += 1.0
		self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_p = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_p = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_p = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_p = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_p.weight.data.normal_(0, 0.5)
		self.itemEmbed_p = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_p.weight.data.normal_(0, 0.5)

		self.globalBias_n = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_n = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_n = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_n = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_n.weight.data.normal_(0, 0.5)
		self.itemEmbed_n = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_n.weight.data.normal_(0, 0.5)

		self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data)*1.5
		self.to(self.device)

	def forward(self, users, items):
		distribution = self.distribution[items].to(self.device)
		price = self.item_price[items].view(-1, 1).expand(users.shape[0], 5).to(self.device)
		reference_point = self.reference_point(users)

		# calculate value
		globalBias_p = self.globalBias_p(torch.tensor(0).to(self.device))
		userBias_p = self.userBias_p(users)
		itemBias_p = self.itemBias_p(items)
		userEmbed_p = self.userEmbed_p(users)
		itemEmbed_p = self.itemEmbed_p(items)

		globalBias_n = self.globalBias_n(torch.tensor(0).to(self.device))
		userBias_n = self.userBias_n(users)
		itemBias_n = self.itemBias_n(items)
		userEmbed_n = self.userEmbed_n(users)
		itemEmbed_n = self.itemEmbed_n(items)

		positive = globalBias_p + userBias_p + itemBias_p + torch.mul(userEmbed_p, itemEmbed_p).sum(1).view(-1, 1)
		negative = globalBias_n + userBias_n + itemBias_n + torch.mul(userEmbed_n, itemEmbed_n).sum(1).view(-1, 1)

		rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[0], 5).to(self.device)
		tanh_r = torch.tanh(rating - reference_point)
		tanh_r_pos = torch.gt(tanh_r, torch.FloatTensor([0]).to(self.device)).to(torch.float)
		tanh_r_neg = torch.ones_like(tanh_r).to(self.device) - tanh_r_pos

		r_pos = torch.mul(tanh_r, tanh_r_pos)
		temp_pos_val = torch.mul(positive, r_pos)
		r_neg = torch.mul(tanh_r, tanh_r_neg)
		temp_neg_val = torch.mul(negative, r_neg)

		value = (temp_pos_val + temp_neg_val).to(torch.float).to(self.device)

		# calculate weight
		globalBias_g = self.globalBias_g(torch.tensor(0).to(self.device))
		userBias_g = self.userBias_g(users)

		globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
		userBias_d = self.userBias_d(users)

		gamma = globalBias_g + userBias_g
		delta = globalBias_d + userBias_d

		nominator = torch.mul(delta, distribution.pow(gamma))
		denominator = torch.mul(delta, distribution.pow(gamma)) + (1.-distribution).pow(gamma)
		weight = torch.div(nominator, denominator)

		return torch.mul(weight, value).sum(1)

	def loss(self, users, items, negItems):
		nusers = users.view(-1, 1).to(self.device)
		nusers = nusers.expand(nusers.shape[0], self.params['negNum_train']).reshape(-1).to(self.device)

		pOut = self.forward(users, items).view(-1, 1).expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
		nOut = self.forward(nusers, negItems).reshape(-1, 1)

		diff = torch.mean(pOut-nOut)
		criterion = nn.Sigmoid()
		loss = criterion(diff)
		return -loss


class TF_plus(nn.Module):
	def __init__(self, userLen, itemLen, distribution, item_price, params):
		super(TF_plus, self).__init__()
		self.userNum = userLen
		self.itemNum = itemLen
		self.params = params

		if 'gpu' in params and params['gpu'] == True:
			self.device = 'cuda'
		else:
			self.device = 'cpu'

		l_size = params['latent_dim']
		self.distribution = torch.FloatTensor(distribution).to(self.device)
		self.item_price = torch.FloatTensor(item_price).to(self.device)

		self.globalBias_g = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_g.weight.data += 1.0
		self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.userBias_g.weight.data.normal_(0, 0.01)

		self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_d.weight.data += 1.0
		self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_t = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_t.weight.data += 1.0
		self.userBias_t = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_p = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_p = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_p = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_p = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_p.weight.data.normal_(0, 0.5)
		self.itemEmbed_p = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_p.weight.data.normal_(0, 0.5)

		self.globalBias_n = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_n = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_n = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_n = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_n.weight.data.normal_(0, 0.5)
		self.itemEmbed_n = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_n.weight.data.normal_(0, 0.5)

		self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data)*2.5
		self.to(self.device)

	def forward(self, users, items):
		distribution = self.distribution[items].to(self.device)
		price = self.item_price[items].view(-1, 1).expand(users.shape[0], 5).to(self.device)
		reference_point = self.reference_point(users)

		# calculate value
		globalBias_p = self.globalBias_p(torch.tensor(0).to(self.device))
		userBias_p = self.userBias_p(users)
		itemBias_p = self.itemBias_p(items)
		userEmbed_p = self.userEmbed_p(users)
		itemEmbed_p = self.itemEmbed_p(items)

		globalBias_n = self.globalBias_n(torch.tensor(0).to(self.device))
		userBias_n = self.userBias_n(users)
		itemBias_n = self.itemBias_n(items)
		userEmbed_n = self.userEmbed_n(users)
		itemEmbed_n = self.itemEmbed_n(items)

		positive = globalBias_p + userBias_p + itemBias_p + torch.mul(userEmbed_p, itemEmbed_p).sum(1).view(-1, 1)
		negative = globalBias_n + userBias_n + itemBias_n + torch.mul(userEmbed_n, itemEmbed_n).sum(1).view(-1, 1)

		rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[0], 5).to(self.device)
		tanh_r = torch.tanh(rating - reference_point)

		tanh_r_pos = torch.gt(tanh_r, torch.FloatTensor([0]).to(self.device)).to(torch.float)
		tanh_r_neg = torch.ones_like(tanh_r).to(self.device) - tanh_r_pos

		r_pos = torch.mul(tanh_r, tanh_r_pos)
		temp_pos_val = torch.mul(positive, r_pos)
		r_neg = torch.mul(tanh_r, tanh_r_neg)
		temp_neg_val = torch.mul(negative, r_neg)

		value = (temp_pos_val + temp_neg_val).to(torch.float).to(self.device)

		# calculate weight
		globalBias_g = self.globalBias_g(torch.tensor(0).to(self.device))
		userBias_g = self.userBias_g(users)

		globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
		userBias_d = self.userBias_d(users)

		globalBias_t = self.globalBias_t(torch.tensor(0).to(self.device))
		userBias_t = self.userBias_t(users)

		gamma = globalBias_g + userBias_g
		delta = globalBias_d + userBias_d
		theta = globalBias_t + userBias_t

		nominator = torch.mul(delta, distribution.pow(gamma))
		# denominator = torch.mul(delta, distribution.pow(gamma)) + (1.-distribution).pow(gamma)
		denominator = (torch.mul(delta, distribution.pow(gamma)) + torch.mul((1.-distribution).pow(gamma), theta)).pow(gamma)
		weight = torch.div(nominator, denominator)

		return torch.mul(weight, value).sum(1)

	def loss(self, users, items, negItems):
		nusers = users.view(-1, 1).to(self.device)
		nusers = nusers.expand(nusers.shape[0], self.params['negNum_train']).reshape(-1).to(self.device)

		pOut = self.forward(users, items).view(-1, 1).expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
		nOut = self.forward(nusers, negItems).reshape(-1, 1)

		diff = torch.mean(pOut-nOut)
		criterion = nn.Sigmoid()
		loss = criterion(diff)
		return -loss

class Prelec(nn.Module):
	def __init__(self, userLen, itemLen, distribution, item_price, params):
		super(Prelec, self).__init__()
		self.userNum = userLen
		self.itemNum = itemLen
		self.params = params

		if 'gpu' in params and params['gpu'] == True:
			self.device = 'cuda'
		else:
			self.device = 'cpu'

		l_size = params['latent_dim']
		self.distribution = torch.FloatTensor(distribution).to(self.device)
		self.item_price = torch.FloatTensor(item_price).to(self.device)

		self.globalBias_g = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_g.weight.data += 0.5
		self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_d.weight.data += 1.
		self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)

		self.globalBias_p = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_p = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_p = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_p = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_p.weight.data.normal_(0, 0.5)
		self.itemEmbed_p = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_p.weight.data.normal_(0, 0.5)

		self.globalBias_n = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_n = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.itemBias_n = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		self.userEmbed_n = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_n.weight.data.normal_(0, 0.5)
		self.itemEmbed_n = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_n.weight.data.normal_(0, 0.5)

		self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data)*1.5
		self.to(self.device)

	def forward(self, users, items):
		distribution = self.distribution[items].to(self.device)
		price = self.item_price[items].view(-1, 1).expand(users.shape[0], 5).to(self.device)
		reference_point = self.reference_point(users)

		# calculate value
		globalBias_p = self.globalBias_p(torch.tensor(0).to(self.device))
		userBias_p = self.userBias_p(users)
		itemBias_p = self.itemBias_p(items)
		userEmbed_p = self.userEmbed_p(users)
		itemEmbed_p = self.itemEmbed_p(items)

		globalBias_n = self.globalBias_n(torch.tensor(0).to(self.device))
		userBias_n = self.userBias_n(users)
		itemBias_n = self.itemBias_n(items)
		userEmbed_n = self.userEmbed_n(users)
		itemEmbed_n = self.itemEmbed_n(items)

		positive = globalBias_p + userBias_p + itemBias_p + torch.mul(userEmbed_p, itemEmbed_p).sum(1).view(-1, 1)
		negative = globalBias_n + userBias_n + itemBias_n + torch.mul(userEmbed_n, itemEmbed_n).sum(1).view(-1, 1)

		rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[0], 5).to(self.device)

		tanh_r = torch.tanh((rating - reference_point)/5.)
		tanh_r_pos = torch.gt(tanh_r, torch.FloatTensor([0]).to(self.device)).to(torch.float)
		tanh_r_neg = torch.ones_like(tanh_r).to(self.device) - tanh_r_pos

		r_pos = torch.mul(tanh_r, tanh_r_pos)
		temp_pos_val = torch.mul(positive, r_pos)
		r_neg = torch.mul(tanh_r, tanh_r_neg)
		temp_neg_val = torch.mul(negative, r_neg)

		value = (temp_pos_val + temp_neg_val).to(torch.float).to(self.device)

		# calculate weight
		globalBias_g = self.globalBias_g(torch.tensor(0).to(self.device))
		userBias_g = self.userBias_g(users)

		globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
		userBias_d = self.userBias_d(users)

		gamma = globalBias_g + userBias_g
		delta = globalBias_d + userBias_d

		weight = torch.exp(-torch.mul(delta, torch.pow(-torch.log(distribution), gamma)))

		return torch.mul(weight, value).sum(1)

	def loss(self, users, items, negItems):
		nusers = users.view(-1, 1).to(self.device)
		nusers = nusers.expand(nusers.shape[0], self.params['negNum_train']).reshape(-1).to(self.device)

		pOut = self.forward(users, items).view(-1, 1).expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
		nOut = self.forward(nusers, negItems).reshape(-1, 1)

		diff = torch.mean(pOut-nOut)
		criterion = nn.Sigmoid()
		loss = criterion(diff)
		return -loss


class Prelec_plus(nn.Module):
	def __init__(self, userLen, itemLen, distribution, item_price, params):
		super(Prelec_plus, self).__init__()
		self.userNum = userLen
		self.itemNum = itemLen
		self.params = params

		if 'gpu' in params and params['gpu'] == True:
			self.device = 'cuda'
		else:
			self.device = 'cpu'

		l_size = params['latent_dim']
		self.distribution = torch.FloatTensor(distribution).to(self.device)
		self.item_price = torch.FloatTensor(item_price).to(self.device)

		self.globalBias_g = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_g.weight.data += 0.5
		self.userBias_g = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		# self.userBias_g.weight.data.normal_(0, 0.01)


		self.globalBias_d = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_d.weight.data += 1.
		self.userBias_d = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		# self.userBias_d.weight.data.normal_(0, 0.01)


		self.globalBias_t = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.globalBias_t.weight.data += 1.
		self.userBias_t = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		# self.userBias_t.weight.data.normal_(0, 0.01)


		self.globalBias_p = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_p = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		# self.userBias_p.weight.data.normal_(0.0, 0.1)
		self.itemBias_p = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		# self.itemBias_p.weight.data.normal_(0.0, 0.1)
		self.userEmbed_p = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_p.weight.data.normal_(0, 0.5)
		self.itemEmbed_p = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_p.weight.data.normal_(0, 0.5)

		self.globalBias_n = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
		self.userBias_n = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		# self.userBias_n.weight.data.normal_(0.0, 0.1)
		self.itemBias_n = ZeroEmbedding(itemLen, 1).to(self.device).to(torch.float)
		# self.itemBias_n.weight.data.normal_(0.0, 0.1)
		self.userEmbed_n = ZeroEmbedding(userLen, l_size).to(self.device).to(torch.float)
		self.userEmbed_n.weight.data.normal_(0, 0.5)
		self.itemEmbed_n = ZeroEmbedding(itemLen, l_size).to(self.device).to(torch.float)
		self.itemEmbed_n.weight.data.normal_(0, 0.5)

		self.reference_point = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
		self.reference_point.weight.data = torch.ones_like(self.reference_point.weight.data)*3.0
		self.to(self.device)

	def forward(self, users, items):
		distribution = self.distribution[items].to(self.device)
		price = self.item_price[items].view(-1, 1).expand(users.shape[0], 5).to(self.device)
		reference_point = self.reference_point(users)

		# calculate value
		globalBias_p = self.globalBias_p(torch.tensor(0).to(self.device))
		userBias_p = self.userBias_p(users)
		itemBias_p = self.itemBias_p(items)
		userEmbed_p = self.userEmbed_p(users)
		itemEmbed_p = self.itemEmbed_p(items)

		globalBias_n = self.globalBias_n(torch.tensor(0).to(self.device))
		userBias_n = self.userBias_n(users)
		itemBias_n = self.itemBias_n(items)
		userEmbed_n = self.userEmbed_n(users)
		itemEmbed_n = self.itemEmbed_n(items)

		positive = globalBias_p + userBias_p + itemBias_p + torch.mul(userEmbed_p, itemEmbed_p).sum(1).view(-1, 1)
		negative = globalBias_n + userBias_n + itemBias_n + torch.mul(userEmbed_n, itemEmbed_n).sum(1).view(-1, 1)

		rating = torch.tensor([1., 2., 3., 4., 5.]).expand(users.shape[0], 5).to(self.device)
		tanh_r = torch.tanh(rating - reference_point)
		tanh_r_pos = torch.gt(tanh_r, torch.FloatTensor([0]).to(self.device)).to(torch.float)
		tanh_r_neg = torch.ones_like(tanh_r).to(self.device) - tanh_r_pos

		r_pos = torch.mul(tanh_r, tanh_r_pos)
		temp_pos_val = torch.mul(positive, r_pos)
		r_neg = torch.mul(tanh_r, tanh_r_neg)
		temp_neg_val = torch.mul(negative, r_neg)

		value = (temp_pos_val + temp_neg_val).to(torch.float).to(self.device)

		# calculate weight
		globalBias_g = self.globalBias_g(torch.tensor(0).to(self.device))
		userBias_g = self.userBias_g(users)

		globalBias_d = self.globalBias_d(torch.tensor(0).to(self.device))
		userBias_d = self.userBias_d(users)

		globalBias_t = self.globalBias_t(torch.tensor(0).to(self.device))
		userBias_t = self.userBias_t(users)

		gamma = globalBias_g + userBias_g
		delta = globalBias_d + userBias_d
		theta = globalBias_t + userBias_t

		weight = torch.mul(torch.exp(-torch.mul(delta, torch.pow(-torch.log(distribution), gamma))), theta)

		return torch.mul(weight, value).sum(1)

	def loss(self, users, items, negItems):
		nusers = users.view(-1, 1).to(self.device)
		nusers = nusers.expand(nusers.shape[0], self.params['negNum_train']).reshape(-1).to(self.device)

		pOut = self.forward(users, items).view(-1, 1).expand(users.shape[0], self.params['negNum_train']).reshape(-1, 1)
		nOut = self.forward(nusers, negItems).reshape(-1, 1)

		diff = torch.mean(pOut-nOut)
		criterion = nn.Sigmoid()
		loss = criterion(diff)
		return -loss