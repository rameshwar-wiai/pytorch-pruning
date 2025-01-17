import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

from tqdm import tqdm

class ModifiedVGG16Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG16Model, self).__init__()

		model = models.vgg16(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Linear(25088, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 10)
			)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

'''Hack to create the new model based on the state dict
'''
class PrunedVGGModel(torch.nn.Module):
	def __init__(self, state_dict):
		# create the new layers based on the state dict
		super(PrunedVGGModel, self).__init__()
		self._make_layers(state_dict)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def _make_layers(self, state_dict):
		feature_layers = []
		classifier_layers = []
		# first make the feature weights
		#print(state_dict.keys())
		prev_index = 0
		for i, layer_name in enumerate(state_dict.keys()):
			# only consider weights, we will consider biases along with them
			if i%2 == 1:
				continue
			print(layer_name)
			layer_type, index, matrix_type = layer_name.strip().split('.')
			index = int(index)
			
			if layer_type == 'features':
				if index - prev_index == 3:
					feature_layers += [nn.MaxPool2d(2, 2)]
				prev_index = index
				output_channels, input_channels, height, width = state_dict[layer_name].shape
				conv2d = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
				conv2d.weight.data = state_dict[layer_name]
				conv2d.bias.data = state_dict['.'.join([layer_type, str(index), 'bias'])]
				feature_layers += [conv2d,  nn.ReLU(inplace=True)]

			elif layer_type == 'classifier':
				print(layer_name, state_dict[layer_name].shape)
				output_dim, input_dim = state_dict[layer_name].shape
				linear_layer = nn.Linear(output_dim, input_dim)
				linear_layer.weight.data = state_dict[layer_name]
				linear_layer.bias.data = state_dict['.'.join([layer_type, str(index), 'bias'])]
				classifier_layers += [linear_layer, nn.ReLU(inplace=True)]
			
			else:
				assert True, "No such layer name: " + layer_type

		# the last MaxPool2d layer
		feature_layers += [nn.MaxPool2d(2, 2)]
		self.features = nn.Sequential(*feature_layers)
		# remove the last ReLU layer
		classifier_layers = classifier_layers[:-1]
		self.classifier = nn.Sequential(*classifier_layers)


class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			x = module(x)
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				x.register_hook(self.compute_rank)
				self.activations.append(x)
				self.activation_to_layer[activation_index] = layer
				activation_index += 1

		return self.model.classifier(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		#print(torch.sum((activation * grad), dim = 0).shape)
		values = torch.sum((activation * grad), dim=0, keepdim = True).sum(dim=2, keepdim = True).sum(dim=3,keepdim = True)[0, :, 0, 0].data
		
		# Normalize the rank by the filter dimensions
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i]).cpu()
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_VGG16:
	def __init__(self, train_path, test_path, model):
		self.train_data_loader = dataset.loader(train_path, batch_size=32)
		self.test_data_loader = dataset.test_loader(test_path, batch_size=32)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

	def test(self):
		self.model.eval()
		correct = 0
		total = 0

		for i, (batch, label) in tqdm(enumerate(self.test_data_loader)):
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
			correct += pred.cpu().eq(label).sum()
			total += label.size(0)
		
		print("Accuracy :", float(correct) / total)
		
		self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.classifier.parameters(), 
					lr=0.001, momentum=0.9)

		for i in range(epoches):
			print("Epoch: ", i)
			self.train_epoch(optimizer)
			self.test()
		print("Finished fine tuning.")
		

	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)

		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in tqdm(self.train_data_loader):
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()

		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)

		print("Number of prunning iterations to reduce 67% filters", iterations)

		for i in range(iterations):
			print("Ranking filters.. Iteration: ", str(i+1) , "/", str(iterations))
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			print("Layers that will be prunned", layers_prunned)
			print("Prunning filters.. ")
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_vgg16_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()

			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print("Filters prunned", str(message))
			self.test()
			print("Fine tuning to recover from prunning iteration.")
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 3)


		print("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, epoches = 3)
		torch.save(model.state_dict(), "model_prunned")

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", dest="train", action="store_true")
	parser.add_argument("--prune", dest="prune", action="store_true")
	parser.add_argument("--train_path", type = str, default = "train")
	parser.add_argument("--test_path", type = str, default = "test")
	parser.set_defaults(train=False)
	parser.set_defaults(prune=False)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()

	if args.train:
		model = ModifiedVGG16Model().cuda()
	elif args.prune:
		model = torch.load("model").cuda()

	fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

	if args.train:
		fine_tuner.train(epoches = 15)
		torch.save(model, "model")

	elif args.prune:
		fine_tuner.prune()
