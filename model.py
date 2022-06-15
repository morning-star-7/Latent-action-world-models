import torch
from torch import nn, pinverse
import torch.nn.functional as F
import numpy as np
from config import config
from tools import get_data_loader, log
import matplotlib.pyplot as plt
from torch import optim
from timm.models.vision_transformer import PatchEmbed, Block
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch.distributed as dist

Tensor = TypeVar('torch.tensor')

class Projector(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Projector, self).__init__()
		self.conv = conv3x3(64, 12)
		self.bn_c = nn.BatchNorm2d(12, momentum=config.bn_momentum)
		
		# only one conv
		
		# self.linear0 = nn.Linear(in_channels, out_channels, bias=False)
		self.linear0 = nn.Linear(12 * config.state_size, out_channels, bias=False)
		self.bn0 = nn.BatchNorm1d(out_channels, momentum=config.bn_momentum)
		self.linear1 = nn.Linear(out_channels, out_channels, bias=False)
		self.bn1 = nn.BatchNorm1d(out_channels, momentum=config.bn_momentum)
		self.act = nn.ReLU()
	
	def forward(self, input):
		x = self.conv(input)
		x = self.bn_c(x)
		x = self.act(x)
		
		x = x.view(-1, 12 * config.state_size)
		
		# x = input
		
		x = self.linear0(x)
		x = self.bn0(x)
		x = self.act(x)
		x = self.linear1(x)
		# x = self.bn1(x)
		# x = self.act(x)
		return x


class Projector2(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Projector2, self).__init__()
		self.in_channels, self.out_channels = in_channels, out_channels
		self.conv1 = conv3x3(self.in_channels, self.in_channels)
		self.bn = nn.BatchNorm2d(self.in_channels, momentum=config.bn_momentum)
		self.act = nn.ReLU()
		self.conv2 = conv3x3(self.in_channels, self.out_channels)
	
	def forward(self, input):
		x = self.conv1(input)
		x = self.bn(x)
		x = self.act(x)
		x = self.conv2(x)
		
		x = x.view(-1, self.out_channels * config.state_size)
		return x


class Predictor(nn.Module):
	def __init__(self):
		super(Predictor, self).__init__()
		self.linear0 = nn.Linear(32, 128)
		self.bn0 = nn.BatchNorm1d(128, momentum=config.bn_momentum)
		self.linear1 = nn.Linear(128, 32)
		self.bn1 = nn.BatchNorm1d(32, momentum=config.bn_momentum)
		self.act = nn.ReLU()
	
	def forward(self, input):
		x = self.linear0(input)
		x = self.bn0(x)
		x = self.act(x)
		x = self.linear1(x)
		# x = self.bn1(x)
		# x = self.act(x)
		return x


def mlp(input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False):
	sizes = [input_size] + layer_sizes + [output_size]
	layers = []
	for i in range(len(sizes) - 1):
		if i < len(sizes) - 2:
			act = activation
			layers += [nn.Linear(sizes[i], sizes[i + 1]),
			           nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
			           act()]
		else:
			act = output_activation
			layers += [nn.Linear(sizes[i], sizes[i + 1]),
			           act()]
	
	if init_zero:
		layers[-2].weight.data.fill_(0)
		layers[-2].bias.data.fill_(0)
	
	return nn.Sequential(*layers)


def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
		super(ResidualBlock, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels, stride)
		self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
		self.downsample = downsample
	
	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = F.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out = F.relu(out)
		return out


class DownSample(nn.Module):
	def __init__(self, in_channels, out_channels, momentum=0.1):
		super(DownSample, self).__init__()
		# self.conv1 = nn.Conv2d(
		# 	in_channels,
		# 	out_channels // 2,
		# 	kernel_size=3,
		# 	stride=2,
		# 	padding=1,
		# 	bias=False,
		# )
		self.conv1 = conv3x3(in_channels, out_channels // 2, stride=2)
		self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
		self.resblocks1 = nn.ModuleList(
			[ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
		)
		self.conv2 = conv3x3(out_channels // 2, out_channels, stride=2)
		self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2,
		                                      downsample=self.conv2)
		self.resblocks2 = nn.ModuleList(
			[ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
		)
		self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
		self.resblocks3 = nn.ModuleList(
			[ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
		)
		self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = nn.functional.relu(x)
		for block in self.resblocks1:
			x = block(x)
		x = self.downsample_block(x)
		for block in self.resblocks2:
			x = block(x)
		x = self.pooling1(x)
		for block in self.resblocks3:
			x = block(x)
		x = self.pooling2(x)
		return x


# observation_shape = [4, 96, 96]
# num_block = 1
# num_channel = 64
# downsample = True
# batch_size  256

class RepresentationNetwork(nn.Module):
	def __init__(
			self,
			observation_shape,
			num_blocks,
			num_channels,
			downsample,
			momentum=0.1):
		super(RepresentationNetwork, self).__init__()
		self.downsample = downsample
		if self.downsample:
			self.downsample_net = DownSample(
				observation_shape[0],
				num_channels,
				momentum=momentum
			)
		self.conv = conv3x3(
			observation_shape[0],
			num_channels,
		)
		self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
		self.resblocks = nn.ModuleList(
			[ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
		)
	
	def forward(self, x):
		if self.downsample:
			x = self.downsample_net(x)
		else:
			x = self.conv(x)
			x = self.bn(x)
			x = nn.functional.relu(x)
		
		for block in self.resblocks:
			x = block(x)
		return x
	
	def get_param_mean(self):
		# TODO: what does this func use for?
		mean = []
		for name, param in self.named_parameters():
			mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
		mean = sum(mean) / len(mean)
		return mean


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		modules = []
		
		hidden_dims = [64*4, 64*4, 64*4, 64*4]
		hidden_dims.reverse()
		
		for i in range(len(hidden_dims) - 1):
			modules.append(
				nn.Sequential(
					nn.ConvTranspose2d(hidden_dims[i],
					                   hidden_dims[i + 1],
					                   kernel_size=3,
					                   stride=2,
					                   padding=1,
					                   output_padding=1),
					nn.BatchNorm2d(hidden_dims[i + 1], momentum=config.bn_momentum),
					nn.ReLU())
			)
		
		self.decoder = nn.Sequential(*modules)
		
		self.final_layer = nn.Sequential(
			nn.ConvTranspose2d(hidden_dims[-1],
			                   hidden_dims[-1],
			                   kernel_size=3,
			                   stride=2,
			                   padding=1,
			                   output_padding=1),
			nn.BatchNorm2d(hidden_dims[-1], momentum=config.bn_momentum),
			nn.ReLU(),
			nn.Conv2d(hidden_dims[-1], out_channels=3,
			          kernel_size=3, padding=1),
			nn.Sigmoid())
	
	def forward(self, z):
		result = self.decoder(z)
		result = self.final_layer(result)
		return result


class VectorQuantizer(nn.Module):
	def __init__(self, num_embeddings, embedding_dim, commitment_cost):
		super(VectorQuantizer, self).__init__()
		
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings
		
		self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
		self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
		self._commitment_cost = commitment_cost
	
	def forward(self, input):
		# convert inputs from BCHW -> BHWC
		# inputs = inputs.permute(0, 2, 3, 1).contiguous()
		input_shape = input.shape
		device = input.device
		
		# Flatten input
		flat_input = input.view(-1, self._embedding_dim)
		
		# print(flat_input.shape)
		# print(torch.sum(flat_input ** 2, dim=1, keepdim=True).shape)
		# print(torch.sum(self._embedding.weight ** 2, dim=1).shape)
		# print((torch.sum(flat_input ** 2, dim=1, keepdim=True)
		#              + torch.sum(self._embedding.weight ** 2, dim=1)).shape)
		
		# Calculate distances
		distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
		             + torch.sum(self._embedding.weight ** 2, dim=1)
		             - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
		
		# Encoding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
		encodings.scatter_(1, encoding_indices, 1)
		
		# Quantize and unflatten
		quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
		
		# Loss
		e_latent_loss = torch.mean((quantized.detach() - input) ** 2)
		q_latent_loss = torch.mean((quantized - input.detach()) ** 2)
		loss = q_latent_loss + self._commitment_cost * e_latent_loss
		
		quantized = input + (quantized - input).detach()
		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		
		# convert quantized from BHWC -> BCHW
		# return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
		return quantized, loss, perplexity, encodings


class VectorQuantizer1D(nn.Module):
	# def __init__(self, num_embeddings, input_sizes, layer_sizes, embedding_dim, commitment_cost):
	def __init__(self, num_embeddings, input_sizes, embedding_dim, commitment_cost):
		super(VectorQuantizer1D, self).__init__()
		
		# mlps = mlp(input_sizes, layer_sizes, embedding_dim)
		self.linear = nn.Linear(input_sizes, embedding_dim)
		self.bn = nn.BatchNorm1d(embedding_dim, affine=False)
		
		self._input_sizes = input_sizes
		self._embedding_dim = embedding_dim
		self._num_embeddings = num_embeddings
		
		self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
		self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
		self._commitment_cost = commitment_cost
	
	def get_embedding(self, index):
		return self._embedding.weight[index]
	
	def forward(self, input):
		# convert inputs from BCHW -> BHWC
		# inputs = inputs.permute(0, 2, 3, 1).contiguous()
		input_shape = input.shape
		device = input.device
		
		# Flatten input
		# TODO multilayer, pooling
		# print('self.input:',self._input_sizes)
		# print('input:',input.shape)
		# quit()
		
		flat_input = input.contiguous().view(-1, self._embedding_dim) # shape []
		# flat_input = self.linear(flat_input)


		# flat_input = self.bn(flat_input)
		
		# print(flat_input.shape)
		# print(torch.sum(flat_input ** 2, dim=1, keepdim=True).shape)
		# print(torch.sum(self._embedding.weight ** 2, dim=1).shape)
		# print((torch.sum(flat_input ** 2, dim=1, keepdim=True)
		#              + torch.sum(self._embedding.weight ** 2, dim=1)).shape)
		
		# Calculate distances
		distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
		             + torch.sum(self._embedding.weight ** 2, dim=1)
		             - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
		# print('distance shape:',distances.shape)

		# print('0',distances.shape)
		# print(flat_input.shape)
		# print('1',torch.sum(flat_input ** 2, dim=1, keepdim=True).shape)
		# print('2',torch.sum(self._embedding.weight ** 2, dim=1).shape)
		# print('3',torch.matmul(flat_input, self._embedding.weight.t()).shape)
		# quit()
		
		# Encoding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		# print('encoding index:',encoding_indices.shape)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
		encodings.scatter_(1, encoding_indices, 1)
		# print('encodings shape:',encodings.shape)
		
		# Quantize and unflatten
		quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)
		quantized=quantized.view(input_shape)
		# print('quantize shape:',quantized.shape)
		
		# Loss
		# e_latent_loss = torch.mean((quantized.detach() - flat_input) ** 2)
		# q_latent_loss = torch.mean((quantized - flat_input.detach()) ** 2)
		e_latent_loss = F.mse_loss(quantized.detach(), input)
		q_latent_loss = F.mse_loss(quantized, input.detach())
		loss = q_latent_loss + self._commitment_cost * e_latent_loss
		
		# quantized = input + (quantized - input).detach()
		# print(flat_input.shape, quantized.shape)
		quantized = input + (quantized - input).detach()
		# quantized=quantized.view(input.shape[0],-1,*input.shape[-1:])
		# print('z shape:',quantized.shape)

		# print(quantized.shape)
		# quit()
		# quantized = quantized.unsqueeze(-1).repeat(1, 1, *input.shape[-1:])

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		
		# convert quantized from BHWC -> BCHW
		# return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
		return quantized, loss, perplexity, encodings



class BottleneckBlock(nn.Module):
	def __init__(self, k_bins, emb_width, mu):
		super().__init__()
		self.k_bins = k_bins
		self.emb_width = emb_width
		self.mu = mu
		self.reset_k()
		self.threshold = 1.0

	def reset_k(self):
		self.init = False
		self.k_sum = None
		self.k_elem = None
		self.register_buffer('k', torch.zeros(self.k_bins, self.emb_width).cuda())

	def _tile(self, x):
		d, ew = x.shape
		if d < self.k_bins:
			n_repeats = (self.k_bins + d - 1) // d
			std = 0.01 / np.sqrt(ew)
			x = x.repeat(n_repeats, 1)
			x = x + torch.randn_like(x) * std
		return x

	def init_k(self, x):
		mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
		self.init = True
		# init k_w using random vectors from x
		y = self._tile(x)
		_k_rand = y[torch.randperm(y.shape[0])][:k_bins]
		dist.broadcast(_k_rand, 0)
		self.k = _k_rand
		assert self.k.shape == (k_bins, emb_width)
		self.k_sum = self.k
		self.k_elem = torch.ones(k_bins, device=self.k.device)

	def restore_k(self, num_tokens=None, threshold=1.0):
		mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
		self.init = True
		assert self.k.shape == (k_bins, emb_width)
		self.k_sum = self.k.clone()
		self.k_elem = torch.ones(k_bins, device=self.k.device)
		if num_tokens is not None:
			expected_usage = num_tokens / k_bins
			self.k_elem.data.mul_(expected_usage)
			self.k_sum.data.mul_(expected_usage)
		self.threshold = threshold

	def update_k(self, x, x_l):
		mu, emb_width, k_bins = self.mu, self.emb_width, self.k_bins
		with torch.no_grad():
			# Calculate new centres
			x_l_onehot = torch.zeros(k_bins, x.shape[0], device=x.device)  # k_bins, N * L
			x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

			_k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
			_k_elem = x_l_onehot.sum(dim=-1)  # k_bins
			y = self._tile(x)
			_k_rand = y[torch.randperm(y.shape[0])][:k_bins]

			dist.broadcast(_k_rand, 0)
			dist.all_reduce(_k_sum)
			dist.all_reduce(_k_elem)

			# Update centres
			old_k = self.k
			self.k_sum = mu * self.k_sum + (1. - mu) * _k_sum  # w, k_bins
			self.k_elem = mu * self.k_elem + (1. - mu) * _k_elem  # k_bins
			usage = (self.k_elem.view(k_bins, 1) >= self.threshold).float()
			self.k = usage * (self.k_sum.view(k_bins, emb_width) / self.k_elem.view(k_bins, 1)) \
					+ (1 - usage) * _k_rand
			_k_prob = _k_elem / torch.sum(_k_elem)  # x_l_onehot.mean(dim=-1)  # prob of each bin
			entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))  # entropy ie how diverse
			used_curr = (_k_elem >= self.threshold).sum()
			usage = torch.sum(usage)
			dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
		return dict(entropy=entropy,
					used_curr=used_curr,
					usage=usage,
					dk=dk)

	def preprocess(self, x):
		# NCT -> NTC -> [NT, C]
		# x = x.permute(0, 2, 1).contiguous()
		# x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)
		x=x.contiguous().view(-1,self.emb_width)

		if x.shape[-1] == self.emb_width:
			prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
		elif x.shape[-1] == 2 * self.emb_width:
			x1, x2 = x[...,:self.emb_width], x[...,self.emb_width:]
			prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape)))

			# Normalise
			x = x1 + x2
		else:
			assert False, f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
		return x, prenorm

	def postprocess(self, x_l, x_d, x_shape):
		# [NT, C] -> NTC -> NCT
		N, T = x_shape
		x_d = x_d.view(N, -1, T).contiguous()
		# x_l = x_l.view(N, T)
		return x_l, x_d

	def quantise(self, x):
		# Calculate latent code x_l
		k_w = self.k.t()
		distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
																							keepdim=True)  # (N * L, b)
		min_distance, x_l = torch.min(distance, dim=-1)
		fit = torch.mean(min_distance)
		return x_l, fit

	def dequantise(self, x_l):
		x = F.embedding(x_l, self.k)
		return x

	def encode(self, x):
		N, width, T = x.shape

		# Preprocess.
		x, prenorm = self.preprocess(x)

		# Quantise
		x_l, fit = self.quantise(x)

		# Postprocess.
		x_l = x_l.view(N, T)
		return x_l

	def decode(self, x_l):
		N, T = x_l.shape
		width = self.emb_width

		# Dequantise
		x_d = self.dequantise(x_l)

		# Postprocess
		x_d = x_d.view(N, T, width).permute(0, 2, 1).contiguous()
		return x_d

	def forward(self, x, update_k=True):
		# print(x.shape)
		# quit()
		N, width, T = x.shape # [4,4,768]

		# Preprocess
		x, prenorm = self.preprocess(x)
		# print(x.shape)
		# quit()

		# Init k if not inited
		if update_k and not self.init:
			self.init_k(x)

		# Quantise and dequantise through bottleneck
		x_l, fit = self.quantise(x)
		x_d = self.dequantise(x_l)

		# Update embeddings
		if update_k:
			update_metrics = self.update_k(x, x_l)
		else:
			update_metrics = {}

		# Loss
		commit_loss = torch.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)
		q_loss = torch.norm(x_d - x.detach()) ** 2 / np.prod(x.shape)
		# loss = q_loss + 0.02*commit_loss
		loss = 0.02*commit_loss

		# Passthrough
		x_d = x + (x_d - x).detach()

		# Postprocess
		x_l, x_d = self.postprocess(x_l, x_d, (N,T))
		# print('x_d shape:',x_d.shape)
		# print('x_l shape:',x_l.shape)
		# quit()
		return x_l, x_d, loss, dict(fit=fit,
										pn=prenorm,
										**update_metrics)



class LatentActionGen(nn.Module):
	def __init__(self, num_embeddings, in_channel, embedding_channel, num_blocks):
		super(LatentActionGen, self).__init__()
		vq_in_channel = 5
		# self.quantizer = VectorQuantizer(num_embeddings, embedding_channel * config.state_size, 0.1)
		self.quantizer = VectorQuantizer1D(num_embeddings, config.latent_dim, embedding_channel, 0.25)
		self.bottleneck=BottleneckBlock(num_embeddings,embedding_channel,0.99)
		self.conv = conv3x3(in_channel * 2, in_channel)
		# self.conv = conv3x3(in_channel * 2, embedding_channel) # sample
		self.bn = nn.BatchNorm2d(in_channel, momentum=config.bn_momentum)
		self.act = nn.ReLU()
		self.resblocks = nn.ModuleList(
			[ResidualBlock(in_channel, in_channel) for _ in range(num_blocks)]
		)
		self.conv_out = conv3x3(in_channel, vq_in_channel)
		self.blocks = nn.ModuleList([
            Block(config.latent_dim, 12, 4, qkv_bias=True,  norm_layer=nn.LayerNorm)
            for i in range(4)])
		self.bottleneck_dim=config.latent_action_channel
		self.linear_down=nn.Linear(config.latent_dim,self.bottleneck_dim)
		self.linear_up=nn.Linear(self.bottleneck_dim,config.latent_dim)
	
	def forward(self, s0, s1, pos_embed_set, latent_diff):
		# add positional embedding
		s1_=s1
		s0=s0+pos_embed_set[:,0:197,:]
		s1=s1+pos_embed_set[:,200:397,:]
		latent_diff=latent_diff+pos_embed_set[:,400:400+latent_diff.shape[1],:]
		s01 = torch.cat([s0, s1,latent_diff], dim=1)
		x=s01
		for block in self.blocks:
			x=block(x)
		x=x[:,-latent_diff.shape[1]:,:]
		x=self.linear_down(x)
		# x=s1_

		# x = self.conv(s01)
		# x = self.bn(x)
		# x = self.act(x)
		# for block in self.resblocks:
		# 	x = block(x)
		# x = self.conv_out(x)
		# z, loss, perplexity, encodings = self.quantizer(x)
		x_l,z,loss,dicts=self.bottleneck(x)
		perplexity=None
		z=self.linear_up(z)
		# z=x
		# print(z.shape)
		# quit()
		# print('perp:', perplexity)
		return z, loss, perplexity


def test_quantizer():
	v = VectorQuantizer(10, 100, 0.01)
	a = torch.randn((32, 4, 5, 5))
	loss, quantized, perplexity, encodings = v(a)
	print(encodings)


class Dynamic(nn.Module):
	def __init__(self, s_channel, z_channel, num_blocks):
		super(Dynamic, self).__init__()
		self.conv = conv3x3(s_channel + z_channel, s_channel)
		self.bn = nn.BatchNorm2d(s_channel, momentum=config.bn_momentum)
		self.act = nn.ReLU()
		self.resblocks = nn.ModuleList(
			[ResidualBlock(s_channel, s_channel) for _ in range(num_blocks)]
		)
		self.blocks = nn.ModuleList([
            Block(config.latent_dim, 12, 4, qkv_bias=True,  norm_layer=nn.LayerNorm)
            for i in range(2)])
	
	def forward(self, s, z, pos_embed_set,produced_latent):
		# add positional embedding
		z_=z
		z_shape=z.shape
		s=s+pos_embed_set[:,0:197,:]
		z=z+pos_embed_set[:,400:400+z_shape[1],:]
		produced_latent=produced_latent+pos_embed_set[:,200:200+197,:]
		sz = torch.cat([s, z, produced_latent], dim=1)
		x=sz
		for block in self.blocks:
			x=block(x) 

		# # if 4 frames stack
		# x=x[:,:197*4,:]
		
		# if single frame
		x=x[:,-197:,:]
		# x=x[:,:197,:]
		# x=z_

		# x = self.conv(sz)
		# x = self.bn(x)
		# x = self.act(x)
		# for block in self.resblocks:
		# 	x = block(x)
		# return x + s
		return x


decoder__ = Decoder()


class Reconstruct(nn.Module):
	def __init__(self, encoder):
		super(Reconstruct, self).__init__()
		self.encoder = RepresentationNetwork(config.observation_shape, num_blocks=1, num_channels=64, downsample=True)
		self.encoder.load_state_dict(encoder.state_dict())
		
		self.decoder = Decoder()
		self.decoder.load_state_dict(decoder__.state_dict())
		
		self.encoder.requires_grad_(False)
		
		self.optim = optim.Adam(self.parameters(), lr=0.001)
	
	def forward(self, s):
		if config.freeze_encoder_stat:
			self.encoder.eval()
		x = self.encoder(s)
		x = self.decoder(x)
		return x
	
	def learn(self, s, visualize=False):
		self.optim.zero_grad()
		self.train()
		_s = self(s)
		s = F.pad(s[:, -1:], (6, 6, 6, 6))
		loss = ((s - _s) ** 2).sum(dim=(2, 3)).mean()
		loss.backward()
		
		if visualize:
			fig, axs = plt.subplots(1, 2, figsize=(5, 5))
			axs[0].imshow(s[0, -1].detach().cpu().numpy(), cmap='gray')
			axs[1].imshow(_s[0, -1].detach().cpu().numpy(), cmap='gray')
			plt.show()
		
		self.optim.step()
		return loss.item()
	
	def test(self, s):
		self.eval()
		with torch.no_grad():
			_s = self(s)
			s = F.pad(s[:, -1:], (6, 6, 6, 6))
			loss = ((s - _s) ** 2).sum(dim=(2, 3)).mean()
		return loss


def state_reconstruct_test(model, dataset):
	recover = Reconstruct(model.encoder).to(config.device)
	data_loader = get_data_loader(dataset)
	loss_avg = 0.
	
	# cnt = 0
	# stop = 600
	# for data in data_loader:
	# 	(obs0, obs1), action, reward = data
	# 	obs0 = obs0.type(torch.float32).to(config.device) / 255
	# 	# loss = recover.learn(obs0, visualize=(cnt % 10 == 0))
	# 	loss = recover.learn(obs0, visualize=False)
	# 	loss_avg = loss_avg * 0.9 + loss * 0.1
	# 	print('@', loss, loss_avg)
	#
	# 	cnt += 1
	# 	if cnt % 100 == 0:
	# 		log('--> recover: %.6f' % loss_avg)
	# 	if cnt >= stop:
	# 		break
	for i in range(10):
		v = True
		for data in data_loader:
			(obs0, obs1), action, reward = data
			obs0 = obs0.type(torch.float32).to(config.device) / 255
			loss = recover.learn(obs0, visualize=v)
			v = False
			loss_avg = loss_avg * 0.95 + loss * 0.05
			print('@', loss)
		log('--> recover: %.6f' % loss_avg)
	log('-------------')


class ActionInit:
	def __init__(self):
		self.conv__ = conv3x3(128, 12)
		self.bn__ = nn.BatchNorm2d(12)
		self.a__ = nn.Linear(12 * config.state_size, 4)


ai = ActionInit()


class Action(nn.Module):
	def __init__(self, encoder, quantizer=None):
		super(Action, self).__init__()
		self.encoder = RepresentationNetwork(config.observation_shape, num_blocks=1, num_channels=64, downsample=True)
		self.quantizer = quantizer
		# self.r = __.clone()
		
		self.conv = conv3x3(128, 12)
		self.bn = nn.BatchNorm2d(12)
		self.a = nn.Linear(12 * config.state_size, 4)
		
		self.encoder.load_state_dict(encoder.state_dict())
		self.conv.load_state_dict(ai.conv__.state_dict())
		self.bn.load_state_dict(ai.bn__.state_dict())
		self.a.load_state_dict(ai.a__.state_dict())
		
		self.act = nn.ReLU()
		self.encoder.requires_grad_(False)
		
		self.optim = optim.Adam(self.parameters(), lr=0.001)
	
	def forward(self, obs0, obs1):
		if config.freeze_encoder_stat:
			self.encoder.eval()
		s0 = self.encoder(obs0)
		s1 = self.encoder(obs1)
		z = self.conv(torch.cat([s0, s1], dim=1))
		z = self.bn(z)
		z = self.act(z)
		z = z.view(-1, 12 * config.state_size)
		a = self.a(z)
		return a
	
	def learn(self, obs0, obs1, action):
		self.optim.zero_grad()
		self.train()
		
		a = self(obs0, obs1)
		loss = F.cross_entropy(a, action.type(torch.long))
		loss.backward()
		# print(self.r.weight)
		# print(self.h.linear0.weight)
		# print('=========================================')
		self.optim.step()
		return loss.item()


def action_regress_test(model, dataset):
	action_prediction = Action(model.encoder).to(config.device)
	data_loader = get_data_loader(dataset)
	for i in range(10):
		loss_list = []
		for (obs0, obs1), action, reward in data_loader:
			# print(data.shape)
			obs0 = obs0.type(torch.float32).to(config.device) / 255
			obs1 = obs1.type(torch.float32).to(config.device) / 255
			loss = action_prediction.learn(obs0, obs1, action.view(-1).to(config.device))
			loss_list.append(loss)
			print('@: %.5f' % loss)
		log('--> regress: %.6f' % np.mean(loss_list))
	log('-------------')
	pass



class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss],self.decode(encoding)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        # return {'loss': loss,
        #         'Reconstruction_Loss': recons_loss,
        #         'VQ_Loss':vq_loss}
        return loss

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
















if __name__ == '__main__':
	test_quantizer()
