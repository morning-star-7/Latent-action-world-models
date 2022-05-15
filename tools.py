import os
import re
import time
import torch
import numpy as np
import torch.nn.functional as F
from config import config
from torch.utils.data import DataLoader
from torchvision import transforms


def check(files, new_file):
	ptn = re.compile('^' + new_file)
	for file in files:
		if ptn.search(file) is not None:
			return False
	return True


class Logger:
	def __init__(self):
		self.model = None
		self.file = None
	
	def set_model(self, model):
		self.model = model
		files = os.listdir('log/')
		cnt = 0
		while True:
			cnt += 1
			file = 'log_%d' % cnt
			if check(files, file):
				self.file = 'log/%s_%s.txt' % (file, self.model)
				break
		self.write_line('Model: %s' % self.model)
		self.write_line('Time: %s' % time.strftime('%Y.%m.%d-%H:%M:%S', time.localtime()))
	
	def write(self, text):
		if self.file is None:
			raise RuntimeError('Logger: have not set model name')
		print('log %s: %s' % (self.file, text))
		logfile = open(self.file, 'a')
		logfile.write(text)
		logfile.close()
	
	def write_line(self, text):
		self.write(text + '\n')
	
	def __call__(self, text):
		self.write_line(text)


def momentum_update(model0, model1, tau=0.95):
	with torch.no_grad():
		dict0 = model0.state_dict()
		dict1 = model1.state_dict()
		# print(dict0)
		
		for name in dict0:
			# print(name)
			dict0[name] = dict0[name] * tau + dict1[name] * (1. - tau)
		# print('==============')
		model0.load_state_dict(dict0)
	# print(dict1)
	# print(model0.state_dict())
	pass


def NT_Xent(out_a, out_b, hidden_norm=True, temperature=1.0):
	if hidden_norm:
		out_a = F.normalize(out_a)
		out_b = F.normalize(out_b)
	batch_size = out_a.shape[0]
	
	INF = 1e9
	labels = torch.arange(batch_size, device=config.device)
	masks = F.one_hot(torch.arange(batch_size, device=config.device), batch_size)
	logits_aa = out_a @ out_a.T / temperature
	logits_bb = out_b @ out_b.T / temperature
	# print(logits_aa)
	logits_aa = logits_aa - masks * INF / temperature
	logits_bb = logits_bb - masks * INF / temperature
	logits_ab = out_a @ out_b.T / temperature
	logits_ba = out_b @ out_a.T / temperature
	loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=-1), labels)
	loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=-1), labels)
	# print(logits_aa)
	# print(loss_a)
	# exit(0)
	return loss_a + loss_b


def bisect(a, x):
	lo, hi = -1, len(a) - 1
	while lo < hi:
		mid = -(-(lo + hi) // 2)
		if a[mid] > x:
			hi = mid - 1
		else:
			lo = mid
	return lo


class AddGaussianNoise:
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
	
	def __call__(self, tensor: torch.Tensor):
		tensor = tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
		return tensor.clip(0., 1.)
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def simsiam_distance(_p, _z):
	_z = _z.detach()
	_p = F.normalize(_p)
	_z = F.normalize(_z)
	return 1. - (_p * _z).sum(dim=1).mean()


def get_data_loader(dataset):
	return DataLoader(dataset,
	                  batch_size=config.batch_size,
	                  drop_last=True,
	                  shuffle=True,
	                  num_workers=5)


def renormalize(tensor, first_dim=1):
	# normalize the tensor (states)
	if first_dim < 0:
		first_dim = len(tensor.shape) + first_dim
	flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
	max = torch.max(flat_tensor, first_dim, keepdim=True).values
	min = torch.min(flat_tensor, first_dim, keepdim=True).values
	flat_tensor = (flat_tensor - min) / (max - min)
	print(max.mean(), min.mean(), max.shape)
	
	return flat_tensor.view(*tensor.shape)


log = Logger()


def log_setting():
	log('--------------')
	log('max_cd: %s' % str(config.max_cd))
	log('zero_cd: %s' % str(config.zero_cd))
	log('batch_size: %s' % str(config.batch_size))
	log('state_norm: %s' % str(config.state_norm))
	log('learning_rate: %s' % str(config.lr))
	log('freeze_encoder_stat: %s' % str(config.freeze_encoder_stat))
	log('representation_loss: %s' % str(config.representation_loss))
	log('latent_action_channel: %s' % str(config.latent_action_channel))
	log('num_embeddings: %s' % str(config.num_embeddings))
	
	# log('s:\t' + str(config.dataset))
	# log('l:\t' + str(config.lr))
	# log('b:\t' + str(config.batch_size))
	# log('c:\t' + str(config.channel))
	# log('m:\t' + str(config.momentum))
	# log('w:\t' + str(config.l2))
	# log('clip:\t' + str(config.clip_max))
	# log('maxT:\t' + str(config.maxT))
	# log('optim:\t' + str(config.optim))
	# log('layers_num:\t' + str(config.layers_num))
	log('--------------')


if __name__ == '__main__':
	log.set_model('RNN')
