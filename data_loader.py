import numpy as np
import math
from scipy.stats import norm
import random
import sys
from tqdm import tqdm

import copy
import torch
from torch.utils.data import Dataset, DataLoader

import time

def read_data(category):
	address = './data/'+category+'/'+category+'_'
	with open(address+'TrainSamples.txt', 'r') as f:
		data = f.readlines()
	TrainSamples = []
	for line in data:
		row = line[:-1].split(',')
		sample=[int(float(i)) for i in row]
		TrainSamples.append(sample)


	with open(address+'ValidationSamples.txt', 'r') as f:
		data = f.readlines()
	ValSamples = []
	for line in data:
		row = line[:-1].split(',')
		sample = [int(float(i)) for i in row]
		ValSamples.append(sample)


	with open(address+'TestSamples.txt', 'r') as f:
		data = f.readlines()
	TestSamples = []
	for line in data:
		row = line[:-1].split(',')
		sample = [int(float(i)) for i in row]
		TestSamples.append(sample)

	return TrainSamples, ValSamples, TestSamples

def approx_Gaussian(frequency):
	distribution = []
	for i in range(len(frequency)):
		mu = 0
		for j in range(5):
			mu += (j+1) * frequency[i][j]
		sigma = 0
		for j in range(5):
			sigma += math.pow(j+1-mu,2) * frequency[i][j]
		if sigma == 0:
			sigma = 0.1
		prob_ij = []
		cdf_ij = []
		for r in range(1,5):
			cdf_ij.append(norm.cdf(r+0.5,mu,sigma))
		prob_ij.append(filter(cdf_ij[0]))
		prob_ij.append(filter(cdf_ij[1]-cdf_ij[0]))
		prob_ij.append(filter(cdf_ij[2]-cdf_ij[1]))
		prob_ij.append(filter(cdf_ij[3]-cdf_ij[2]))
		prob_ij.append(filter(1 - cdf_ij[3]))
		distribution.append(prob_ij)
	return np.array(distribution)

def filter(prob):
	if prob <= 1e-4:
		return 1e-4
	elif prob >= 1-1e-4:
		return 1-1e-4
	else:
		return prob

def get_decumulative(distribution):
	decumulative = [[1.0] for i in range(distribution.shape[0])]
	# decumulative = copy.deepcopy(cumulative)
	for i in range(distribution.shape[0]):
		distribution_i = distribution[i]
		# print('distribution', distribution_i)
		# decumulative[i].append(1.0)
		for j in range(1, 6):
			summation = sum(distribution_i[:j])
			if summation >= 1.:
				decumulative[i].append(1e-10)
			elif summation <= 1e-10:
				decumulative[i].append(1.0)
			else:
				decumulative[i].append(1.-summation)
	return np.array(decumulative)

def get_datasize(category):
	address = "./data/" + category + "/" + category + "_" + "AllSamples.txt"
	AllSamples = list()
	with open(address,'r') as f:
		data = f.readlines()
	for line in data:
		row = line.split(',')
		samples = [int(float(i)) for i in row]
		AllSamples.append(samples)
	all_data = np.array(AllSamples)
	userNum = len(np.unique(all_data[:,0]))
	itemNum = len(np.unique(all_data[:,1]))
	return userNum, itemNum

def get_price(category):
	address = "./data/" + category + "/" + category + "_" + "item_price.npy"
	price = np.load(address)
	return price

def get_distribution(category):
	address = "./data/" + category + "/" + category + "_" + "ItemResult.npy"
	distribution = np.load(address)
	return distribution


class TransactionData(Dataset):
	def __init__(self, transactions, userNum, itemNum, rating_distribution, negs):
		super(TransactionData, self).__init__()
		self.transactions = transactions
		self.L = len(transactions)
		self.users = np.unique(np.array(transactions)[:, 0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.itemset = [i for i in range(self.itemNum)]
		self.itemset = set(self.itemset)
		self.rating_distribution = rating_distribution
		self.userHist = [[] for i in range(self.userNum)]
		for row in transactions:
			self.userHist[row[0]].append(row[1])

		# build neg pools
		print('build train negatives pool')
		self.neg_pools = []
		pbar = tqdm(total=self.userNum)
		for user in range(self.userNum):
			neg_pool = list(random.sample(self.itemset-set(self.userHist[user]), negs))
			self.neg_pools.append(neg_pool)
			pbar.update(1)
		pbar.close()

	def __len__(self):
		return self.L

	def __getitem__(self, idx):
		row = self.transactions[idx]
		user = row[0]
		item = row[1]
		rating = row[2]
		negItem = self.get_neg(user, item)
		distribution = self.rating_distribution[item]
		return {'user': np.array(user).astype(np.int64),
				'item': np.array(item).astype(np.int64),
				'r_distribution': np.array(distribution).astype(float),
				'rating': np.array(rating).astype(float),
				'negItem': np.array(negItem).astype(np.int64)
				}

	def get_neg(self, userid, itemid):
		neg = list(random.sample(self.neg_pools[userid], self.negNum))

		return neg


	def set_negN(self, n):
		if n < 1:
			return
		self.negNum = n

class UserTransactionData(Dataset):
	def __init__(self, transactions, userNum, itemNum, item_price, trainHist):
		super(UserTransactionData, self).__init__()
		self.transactions = transactions
		self.L = userNum
		self.user = np.unique(np.array(transactions)[:, 0])
		self.userNum = userNum
		self.itemNum = itemNum
		self.negNum = 2 #place holder
		self.userHist = [[] for i in range(self.userNum)]
		self.trainHist = trainHist
		self.item_price = item_price
		for row in self.transactions:
			self.userHist[row[0]].append(row[1])

	def __len__(self):
		return self.L


	def __getitem__(self, idx):
		user = self.user[idx]
		posItem = self.userHist[idx]
		posPrice = []
		for i in posItem:
			posPrice.append(self.item_price[i])

		negPrice = []
		negItem = self.get_neg(idx)
		for i in negItem:
			negPrice.append(self.item_price[i])

		budget = self.get_budget(idx)

		return {'user': np.array(user).astype(np.int64),
				'budget': np.array(budget).astype(float),
				'posItem': np.array(posItem).astype(np.int64),
				'posPrice': np.array(posPrice).astype(float),
				'negPrice': np.array(negPrice).astype(float),
				'negItem': np.array(negItem).astype(np.int64)
				}

	def get_neg(self, userId):
		hist = self.userHist[userId] + self.trainHist[userId]
		neg = []
		for i in range(self.negNum):
			while True:
				negId = np.random.randint(self.itemNum)
				if negId not in hist and negId not in neg:
					neg.append(negId)
					break
		return neg

	def set_negN(self, n):
		if n < 1:
			return
		self.negNum = n

	def get_budget(self, userId):
		price = []
		for i in self.trainHist[userId]:
			price = self.item_price[i]
		budget = np.max(np.array(price))
		return budget


