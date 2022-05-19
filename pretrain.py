import pickle
from pickletools import optimize
from pyexpat import model
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tools import momentum_update
from config import config
import numpy as np
from tools import log, log_setting, NT_Xent, renormalize, simsiam_distance, get_data_loader
import torch.nn.functional as F
from atari import AtariDataset
from ssv2 import ssv2Dataset
from model import RepresentationNetwork
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tools import AddGaussianNoise
from model import Projector, Projector2, Decoder, LatentActionGen, Dynamic, conv3x3
from transform import Transforms
import os
import random
from test import prepare_model, show_image
from util.pos_embed import get_2d_sincos_pos_embed
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


pyplot_cnt = 0
# imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
# imagenet_std = torch.tensor([0.229, 0.224, 0.225])
# imagenet_mean=imagenet_mean.to(config.device)
# imagenet_std=imagenet_std.to(config.device)

# set random seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

# setup_seed(666)



class Model(nn.Module):
	def __init__(self, name='naive', num_channels=768, transform=None):
		super(Model, self).__init__()
		self.name = name
		self.encoder = RepresentationNetwork(config.ss_observation_shape,
		                                     num_blocks=5,
		                                     num_channels=num_channels,
		                                     downsample=True,
		                                     momentum=config.bn_momentum)
		self.decoder = Decoder()
		self.model_mae= prepare_model(chkpt_dir='./mae_visualize_vit_base.pth', arch='mae_vit_base_patch16',device=config.device)
		self.model_mae.requires_grad_(False)
		self.lag = LatentActionGen(config.num_embeddings,
		                           num_channels,
		                           config.latent_action_channel,
		                           num_blocks=5)
		self.dynamic = Dynamic(num_channels, config.latent_action_channel, num_blocks=5)
		self.projector = Projector(num_channels, 10)
		# self.predictor = Predictor()
		
		self.num_channels = num_channels
		self.transform = transform
		
		self.optim = optim.Optimizer(self.parameters(), {})
		self.loss = nn.CosineSimilarity()
		
		# Atari
		self.proj_hid = 1024
		self.proj_out = 1024
		self.pred_hid = 512
		self.pred_out = 1024
		
		# # default
		# self.proj_hid = 256
		# self.proj_out = 256
		# self.pred_hid = 64
		# self.pred_out = 256
		
		# TODO bias and affine ?
		self.projection_in_dim = num_channels * config.state_size
		self.projection = nn.Sequential(
			nn.Linear(self.projection_in_dim, self.proj_hid, bias=False),
			nn.BatchNorm1d(self.proj_hid),
			nn.ReLU(),
			nn.Linear(self.proj_hid, self.proj_hid, bias=False),
			nn.BatchNorm1d(self.proj_hid),
			nn.ReLU(),
			nn.Linear(self.proj_hid, self.proj_out),
			nn.BatchNorm1d(self.proj_out, affine=False)
		)
		self.projection_head = nn.Sequential(
			nn.Linear(self.proj_out, self.pred_hid, bias=False),
			nn.BatchNorm1d(self.pred_hid),
			nn.ReLU(),
			nn.Linear(self.pred_hid, self.pred_out),
		)
		self.pos_embed_set = nn.Parameter(torch.zeros(1, 4*196 + 1, 768), requires_grad=False)  # fixed sin-cos embedding
		self.initial_weight()

	def initial_weight(self):
		pos_embed_set = get_2d_sincos_pos_embed(768, int(28), cls_token=True)
		self.pos_embed_set.data.copy_(torch.from_numpy(pos_embed_set).float().unsqueeze(0))		

	def set_optimizer(self, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay):
		if config.optim is optim.SGD:
			self.optim = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
		elif config.optim is optim.Adam:
			self.optim = optim.Adam(self.parameters(), lr=lr)
		else:
			raise NotImplementedError(str(config.optim))
	
	def visualize_embedding(self, obs0, obs1):
		self.eval()
		state_dict = self.decoder.state_dict()
		with torch.no_grad():
			self.decoder.train()
			s0 = self.encoder(obs0)
			_obs0 = self.decoder(s0)
			
			for i in range(config.num_embeddings):
				z = self.lag.quantizer.get_embedding(i)
				z = z.unsqueeze(0).repeat(obs0.shape[0], 1)
				z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, *s0.shape[-2:])
				
				_s1 = self.dynamic(s0, z)
				self.decoder.load_state_dict(state_dict)
				self.decoder.eval()
				_obs1 = self.decoder(_s1)
				
				fig, axs = plt.subplots(2, 2, figsize=(10, 10))
				axs[0, 0].imshow(F.pad(obs0[0, -3:], (6, 6, 6, 6)).T.detach().cpu().numpy())
				axs[0, 1].imshow(F.pad(obs1[0, -3:], (6, 6, 6, 6)).T.detach().cpu().numpy())
				axs[1, 0].imshow(_obs0[0, -3:].T.detach().cpu().numpy())
				axs[1, 1].imshow(_obs1[0, -3:].T.detach().cpu().numpy())
				# print(_obs1[0, -3:].shape)
				# print(_obs1.shape)
				# print(obs1.shape)
				# quit()
				plt.show()
				fig.savefig('visualize_emb/index_%d.png' % i)
				plt.close()
				
				loss_func = nn.BCELoss(reduction='none')
				obs0_pad = F.pad(obs0[:, -3:], (6, 6, 6, 6))
				obs1_pad = F.pad(obs1[:, -3:], (6, 6, 6, 6))
				# loss_0 = (loss_func(_obs0, obs0_pad) - loss_func(obs0_pad, obs0_pad)).sum(dim=(2, 3)).mean()
				# loss_1 = (loss_func(_obs1, obs1_pad) - loss_func(obs1_pad, obs1_pad)).sum(dim=(2, 3)).mean()
				
				loss_0 = (loss_func(_obs0, obs0_pad) - loss_func(obs0_pad, obs0_pad))[0].sum()
				loss_1 = (loss_func(_obs1, obs1_pad) - loss_func(obs1_pad, obs1_pad))[0].sum()
				print('######## %.5f %.5f' % (loss_0, loss_1))
	
	def learn(self, obs0, obs1, visual=False):	
		self.optim.zero_grad()
		self.train()
		
		representation_loss = config.representation_loss
		
		if representation_loss == 'contrastive':
			if self.transform is not None:
				obs_cp = obs0
				obs0 = self.transform(obs0)
				obs1 = self.transform(obs1)
				
				global pyplot_cnt
				
				if pyplot_cnt % 100 == 0:
					# fig, axs = plt.subplots(1, 2, figsize=(5, 5))
					# axs[0].imshow(obs_cp[0, -1].detach().cpu().numpy(), cmap='gray')
					# axs[1].imshow(obs0[0, -1].detach().cpu().numpy(), cmap='gray')
					# plt.show()
					pass
				pyplot_cnt += 1
			s0 = self.encoder(obs1)
			s1 = self.encoder(obs1)
			# h0 = h0.view(config.batch_size, 64 * 6 * 6)
			# h1 = h1.view(config.batch_size, 64 * 6 * 6)
			
			if config.state_norm:
				s0 = renormalize(s0)
				s1 = renormalize(s1)
			
			y0 = self.projector(s0)
			y1 = self.projector(s1)
			
			# loss = nn.CosineSimilarity()(y0, y1)
			loss_repr = NT_Xent(y0, y1, temperature=0.2)
		
		elif representation_loss == 'auto-encoder':
			s0 = self.encoder(obs0)
			s1 = self.encoder(obs1)
			_obs0 = self.decoder(s0)
			
			# if visual:
			# 	fig, axs = plt.subplots(1, 2, figsize=(5, 5))
			# 	axs[0].imshow(obs0[0, -1].detach().cpu().numpy(), cmap='gray')
			# 	axs[1].imshow(_obs0[0, -1].detach().cpu().numpy(), cmap='gray')
			# 	plt.show()
			loss_func = nn.BCELoss(reduction='none')
			# loss_repr = ((F.pad(obs0[:, -1:], (6, 6, 6, 6)) - _obs0) ** 2).sum(dim=(2, 3)).mean()
			obs0_pad = F.pad(obs0[:, -1:], (6, 6, 6, 6))
			loss_repr = (loss_func(_obs0, obs0_pad) - loss_func(obs0_pad, obs0_pad)).sum(dim=(2, 3)).mean()
		
		elif representation_loss == 'SimSiam':
			if self.transform is not None:
				# obs0 = self.transform(F.pad(obs0, (6, 6, 6, 6)))
				# obs1 = self.transform(F.pad(obs1, (6, 6, 6, 6)))
				_obs0 = obs0
				
				obs0 = self.transform(obs0)
				obs1 = self.transform(obs1)
			
			# print(_obs0.shape, obs0.shape)
			
			# if visual:
			# 	fig, axs = plt.subplots(1, 2, figsize=(5, 5))
			# 	axs[0].imshow(obs0[0, -1].detach().cpu().numpy(), cmap='gray')
			# 	axs[1].imshow(_obs0[0, -1].detach().cpu().numpy(), cmap='gray')
			# 	plt.show()
			
			s0 = self.encoder(obs0)
			s1 = self.encoder(obs1)
			
			if config.state_norm:
				s0 = renormalize(s0)
				s1 = renormalize(s1)
			
			z0 = self.projection(s0.view(config.batch_size, self.projection_in_dim))
			z1 = self.projection(s1.view(config.batch_size, self.projection_in_dim))
			p0 = self.projection_head(z0)
			p1 = self.projection_head(z1)
			
			loss_repr = (simsiam_distance(p0, z1) + simsiam_distance(p1, z0)) / 2
		
		else:
			raise NotImplementedError()
		
		dynamic = True
		loss_dyna = 0.
		loss_lag = 0.
		
		if dynamic:
			if config.state_detach:
				s0 = s0.detach()
				s1 = s1.detach()
			
			z, loss_lag, perp = self.lag(s0, s1)
			_s1 = self.dynamic(s0, z)
			# loss_dyna = (((s1 - _s1) ** 2).sum(dim=1)).clip(min=1e-6).sqrt().mean()
			loss_dyna = (((s1 - _s1) ** 2).sum(dim=1)).sqrt().mean()
			
			_obs1 = self.decoder(_s1)
			obs1_pad = F.pad(obs1[:, -1:], (6, 6, 6, 6))
			# loss_repr_dyn = ((F.pad(obs1[:, -1:], (6, 6, 6, 6)) - _obs1) ** 2).sum(dim=(2, 3)).mean()
			loss_func = nn.BCELoss(reduction='none')
			loss_repr_dyn = (loss_func(_obs1, obs1_pad) - loss_func(obs1_pad, obs1_pad)).sum(dim=(2, 3)).mean()
			print('repr_dyn: %.5f' % loss_repr_dyn)
			loss_repr += loss_repr_dyn
			
			if visual:
				with torch.no_grad():
					# print(F.pad(obs0[:, -1], (6, 6, 6, 6)).shape, _obs1.shape)
					fig, axs = plt.subplots(2, 2, figsize=(10, 10))
					axs[0, 0].imshow(obs0_pad[0, -1].cpu().numpy(), cmap='gray')
					axs[0, 1].imshow(obs1_pad[0, -1].cpu().numpy(), cmap='gray')
					axs[1, 0].imshow(_obs0[0, -1].detach().cpu().numpy(), cmap='gray')
					axs[1, 1].imshow(_obs1[0, -1].detach().cpu().numpy(), cmap='gray')
					plt.show()
					plt.close()
		# print(s1.mean())
		# print(_s1.mean())
		print('%.5f %.5f %.5f' % (loss_repr, loss_dyna, loss_lag))
		
		# if perp.item() > 2.0:
		# 	loss = loss_repr + loss_dyna + loss_lag * (perp.item() - 2.)
		# else:
		# 	loss = loss_repr + loss_dyna + loss_lag * (perp.item() - 2.)
		# loss = loss_repr + loss_dyna + loss_lag * perp.item()  # ?
		loss = loss_repr + loss_dyna + loss_lag
		# loss = loss_repr
		
		loss.backward()
		# total_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
		# print(total_norm)
		
		# for p in self.parameters():
		# 	total_norm = nn.utils.clip_grad_norm_(p, max_norm=1.0)
		# 	if total_norm.item() != 0.:
		# 		print(total_norm)
		
		for p in [self.encoder.parameters(),
		          self.decoder.parameters(),
		          self.lag.parameters(),
		          self.dynamic.parameters()]:
			total_norm = nn.utils.clip_grad_norm_(p, max_norm=1.0)
			print('grad_norm:', total_norm)
		self.optim.step()
		
		return loss.item()
	
	def mae_encoder_forward(self,x):
		# if single image
		s,mask, ids_restore=self.model_mae.forward_encoder(x.float(),mask_ratio=0)
		# if 4 frame stack
		# for i in range(4):
		# 	s,mask, ids_restore=self.model_mae.forward_encoder(x[:,i*3:i*3+3].float(),mask_ratio=0)
		# 	if i==0:
		# 		s_4=s
		# 		ids_restore_4=ids_restore
		# 	else:
		# 		s_4=torch.cat([s_4,s],dim=1)
		# 		ids_restore_4=torch.cat([ids_restore_4,ids_restore],dim=1)
		# return s_4,mask, ids_restore_4
		return s,mask, ids_restore
	
	def mae_decoder_forward(self,x,ids_restore):
		x=self.model_mae.forward_decoder(x,ids_restore)
		return x

	def visualize(self,obs0,_obs0,obs1,_obs1):
		# s0,mask0,ids_restore0=self.mae_encoder_forward(obs0)
		# _obs0=self.mae_decoder_forward(s0,ids_restore0)
		# _obs0=self.model_mae.unpatchify(_obs0)
		_obs0_ = torch.einsum('nchw->nhwc', _obs0).detach().cpu()
		obs0_ = torch.einsum('nchw->nhwc', obs0[:, -3:]).detach().cpu()
		_obs1_ = torch.einsum('nchw->nhwc', _obs1).detach().cpu()
		obs1_ = torch.einsum('nchw->nhwc', obs1[:, -3:]).detach().cpu()
		plt.rcParams['figure.figsize'] = [24, 24]
		plt.subplot(1, 4, 1)
		show_image(obs0_[0], "obs_0")
		plt.subplot(1, 4, 2)
		show_image(_obs0_[0], "recon_0")
		plt.subplot(1, 4, 3)
		show_image(obs1_[0], "obs_1")
		plt.subplot(1, 4, 4)
		show_image(_obs1_[0], "recon_1")
		plt.show()
		plt.savefig('test_.png')


	def forward(self,obs0,obs1):

		# auto encoder loss
		s0,mask0,ids_restore0=self.mae_encoder_forward(obs0)
		s1,mask1,ids_restore1=self.mae_encoder_forward(obs1)
		# # if 4 frames stack
		# _obs0=self.mae_decoder_forward(s0[:,-197:,:],ids_restore0[:,-196:])
		# _obs0=self.model_mae.unpatchify(_obs0)

		# if single frame
		_obs0=self.mae_decoder_forward(s0,ids_restore0)
		_obs0=self.model_mae.unpatchify(_obs0)


		# s0 = self.encoder(obs0)
		# s1 = self.encoder(obs1)
		# _obs0 = self.decoder(s0)

		# dynamic loss
		dynamic = True
		loss_dyna = 0.
		loss_lag = 0.
		
		if dynamic:
			if config.state_detach:
				s0 = s0.detach()
				s1 = s1.detach()
			
			z, loss_lag, perp = self.lag(s0, s1, self.pos_embed_set)
			_s1 = self.dynamic(s0, z, self.pos_embed_set)


			# # if 4 frames stack
			# _obs1=self.mae_decoder_forward(_s1[:,-197:,:],ids_restore1[:,-196:])
			# _obs1=self.model_mae.unpatchify(_obs1)
			# if single frame
			_obs1=self.mae_decoder_forward(_s1,ids_restore1)
			_obs1=self.model_mae.unpatchify(_obs1)			
		return obs0, obs1, _obs0, _obs1, s1, _s1, loss_lag

	def calculate_loss(self, obs0, obs1, _obs0, _obs1, s1, _s1, loss_lag):
		# representation loss
		# loss_func = nn.BCELoss(reduction='none')
		loss_func = nn.MSELoss(reduction='none')
		obs0_pad = obs0[:, -3:]
		# print(obs0[10,1,:,:])
		# print(_obs0[10,1,:,:])
		# quit()
		loss_repr = (loss_func(_obs0, obs0[:, -3:]) - loss_func(obs0_pad, obs0_pad)).sum(dim=(2, 3)).mean()
		# print(loss_repr)
		# quit()

		# dynamic loss
		obs1_pad =obs1[:, -3:]
		# loss_repr_dyn = (loss_func(_obs1, obs1[:, -3:]) - loss_func(obs1_pad, obs1_pad)).sum(dim=(1,2, 3)).mean()
		loss_repr_dyn =F.mse_loss(_obs1,obs1)
		# loss_dyna = (((s1 - _s1) ** 2).sum(dim=1)).sqrt().mean()
		# loss_dyna = (((s1 - _s1) ** 2).sum(dim=(1,2))).sqrt().mean()
		loss_dyna = F.mse_loss(s1,_s1)


		# total loss
		loss =  loss_lag + loss_repr_dyn
		# loss=loss_lag
		print('%.5f %.5f %.5f' % (loss_repr_dyn, loss_dyna, loss_lag))
		return loss

	
	def save(self, file_name=''):
		if not file_name:
			file_name = self.name
		if not os.path.exists('save/%s' % file_name):
			os.mkdir('save/%s' % file_name)
		torch.save(self.encoder.state_dict(), 'save/%s/representation.pkl' % file_name)
		torch.save(self.decoder.state_dict(), 'save/%s/decoder.pkl' % file_name)
		torch.save(self.dynamic.state_dict(), 'save/%s/dynamic.pkl' % file_name)
		torch.save(self.lag.state_dict(), 'save/%s/lag.pkl' % file_name)
	
	def restore(self, file_name=''):
		if not file_name:
			file_name = self.name
		if not os.path.exists('save/%s' % file_name):
			raise FileNotFoundError('restore(): can not find file.')
		try:
			self.encoder.load_state_dict(torch.load('save/%s/representation.pkl' % file_name))
		except Exception as e:
			print(e)
		
		try:
			self.decoder.load_state_dict(torch.load('save/%s/decoder.pkl' % file_name))
		except Exception as e:
			print(e)
		
		try:
			self.lag.load_state_dict(torch.load('save/%s/lag.pkl' % file_name))
		except Exception as e:
			print(e)
		
		try:
			self.dynamic.load_state_dict(torch.load('save/%s/dynamic.pkl' % file_name))
		except Exception as e:
			print(e)


# mean teacher (momentum param)
# stop gradient
# predictor

def set_optimizer(lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay):
	if config.optim is optim.SGD:
		optimizer = optim.SGD(model. parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif config.optim is optim.Adam:
		optimizer = optim.Adam(model. parameters(), lr=lr)
	else:
		raise NotImplementedError(str(config.optim))


def train_epoch(model, dataset, optimizer,sampler):
	model.train()
	data_loader = get_data_loader(dataset,sampler)
	cnt = 0
	for data in data_loader:
		print(cnt)
		# print(data.shape)
		# (obs0, obs1), action, reward = data
		obs0, obs1 =data
		obs0 = obs0.type(torch.float32).to(config.device) / 255
		obs1 = obs1.type(torch.float32).to(config.device) / 255


		# normalize
		# obs0 = torch.einsum('nchw->nhwc', obs0)
		# obs1 = torch.einsum('nchw->nhwc', obs1)
		# obs0=obs0-imagenet_mean
		# obs0=obs0/imagenet_std
		# obs1=obs1-imagenet_mean
		# obs1=obs1/imagenet_std
		# obs0 = torch.einsum('nhwc->nchw', obs0)
		# obs1 = torch.einsum('nhwc->nchw', obs1)

		
		# model. optim.zero_grad()
		optimizer.zero_grad()
		


		cnt += 1
		# loss = model.learn(obs0, obs1, visual=(cnt % 50 == 0))
		# print('#', loss)
		obs0, obs1, _obs0, _obs1, s1, _s1, loss_lag=model(obs0, obs1)
		loss=model.module.calculate_loss(obs0, obs1, _obs0, _obs1, s1, _s1, loss_lag)
		loss.mean().backward()
		for p in [
			model.module.lag.parameters(),
			model.module.dynamic.parameters()]:
			total_norm = nn.utils.clip_grad_norm_(p, max_norm=1.0)
			# print('grad_norm:', total_norm)
		optimizer.step()
		if cnt % 20 ==0:
			model.module.visualize(obs0,_obs0,obs1,_obs1)
			# model.module.visualize_embedding(obs0, obs1)
			# break
		print('##', loss.mean().item())
		


def get_train_dataset(subdir, block_id):
	dataset = AtariDataset(subdir, block_id)
	return dataset


def get_tune_dataset():
	dataset = AtariDataset(5, 30, 20000)
	return dataset


def pretrain():

	# distributed training initialize
	torch.distributed.init_process_group(backend="nccl")
	local_rank=torch.distributed.get_rank()
	torch.cuda.set_device(local_rank)
	config.device= torch.device("cuda",local_rank)
	# print(local_rank)
	# print(config.device)
	# quit()

	row_image_transform = transforms.Compose([
		transforms.CenterCrop(224)
	])
	transform = Transforms()
	model = Model('ssae', transform=transform)
	# chkpt_dir = './mae_visualize_vit_base.pth'
	# model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')
	# print('Model loaded.')
	# for name,parameters in model.named_parameters():
	# 	print(name,':',parameters.size())
	# quit()

	# # if nn.Dataparallel
	# if torch.cuda.device_count() > 1:
	# 	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# 	model = nn.DataParallel(model) # device_ids=[0]

	model.to(config.device)
	model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
	model=torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=True, find_unused_parameters=True)
	# broadcast_buffers=False ???????
	model.module.set_optimizer()
	# set_optimizer()
	optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
	
	# model.restore()
	model.module.save()
	# exit(0)
	
	log.set_model(model.module.name)
	log_setting()

	
	# tune_dataset = get_tune_dataset()
	# # state_reconstruct_test(model, tune_dataset)
	# action_regress_test(model, tune_dataset)
	# del tune_dataset
	config.freeze_encoder_stat = True
	
	cnt = 0
	subdir, block_id = 1, 25
	# lr_schedule = [0.0001, 0.001, 0.01, 0.0333, 0.0666, 0.1, 0.2, 0.4, 0.8, 1.0]

	train_dataset = ssv2Dataset(image_path='/home/chc/dataset/ssv2_extracted_frames_5',transform=row_image_transform,cut=None)
	sampler=DistributedSampler(train_dataset)
	while True:
		# if subdir == 1 and block_id < len(lr_schedule):
		# 	model.set_optimizer(config.lr * lr_schedule[block_id])
		
		# train_dataset = get_train_dataset(subdir, block_id)
		# train_dataset = ssv2Dataset(image_path='/home/chc/dataset/ssv2_extracted_frames_5',transform=row_image_transform,cut=None)
		train_epoch(model, train_dataset,optimizer,sampler)
		# del train_dataset
		
		model.module.save()
		
		# tune_dataset = get_tune_dataset()
		# # state_reconstruct_test(model, tune_dataset)
		# action_regress_test(model, tune_dataset)
		# del tune_dataset
		
		block_id += 1
		if block_id == 50:
			subdir += 1
			block_id = 0
		if subdir == 5:
			subdir = 1


if __name__ == '__main__':
	pretrain()

# torchrun --nproc_per_node=8 pretrain.py