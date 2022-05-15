import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import config
from tools import get_data_loader, log
import matplotlib.pyplot as plt
from torch import optim
from timm.models.vision_transformer import PatchEmbed, Block

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
		
		flat_input = input.view(-1, self._input_sizes)
		flat_input = self.linear(flat_input)

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

		# print('0',distances.shape)
		# print(flat_input.shape)
		# print('1',torch.sum(flat_input ** 2, dim=1, keepdim=True).shape)
		# print('2',torch.sum(self._embedding.weight ** 2, dim=1).shape)
		# print('3',torch.matmul(flat_input, self._embedding.weight.t()).shape)
		# quit()
		
		# Encoding
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
		encodings.scatter_(1, encoding_indices, 1)
		
		# Quantize and unflatten
		quantized = torch.matmul(encodings, self._embedding.weight)  # .view(input_shape)
		
		# Loss
		e_latent_loss = torch.mean((quantized.detach() - flat_input) ** 2)
		q_latent_loss = torch.mean((quantized - flat_input.detach()) ** 2)
		loss = q_latent_loss + self._commitment_cost * e_latent_loss
		
		# quantized = input + (quantized - input).detach()
		# print(flat_input.shape, quantized.shape)
		quantized = flat_input + (quantized - flat_input).detach()
		quantized = quantized.unsqueeze(-1).repeat(1, 1, *input.shape[-1:])

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
		
		# convert quantized from BHWC -> BCHW
		# return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
		return quantized, loss, perplexity, encodings


class LatentActionGen(nn.Module):
	def __init__(self, num_embeddings, in_channel, embedding_channel, num_blocks):
		super(LatentActionGen, self).__init__()
		vq_in_channel = 5
		# self.quantizer = VectorQuantizer(num_embeddings, embedding_channel * config.state_size, 0.1)
		self.quantizer = VectorQuantizer1D(num_embeddings, 768, embedding_channel, 0.1)
		self.conv = conv3x3(in_channel * 2, in_channel)
		# self.conv = conv3x3(in_channel * 2, embedding_channel) # sample
		self.bn = nn.BatchNorm2d(in_channel, momentum=config.bn_momentum)
		self.act = nn.ReLU()
		self.resblocks = nn.ModuleList(
			[ResidualBlock(in_channel, in_channel) for _ in range(num_blocks)]
		)
		self.conv_out = conv3x3(in_channel, vq_in_channel)
		self.blocks = nn.ModuleList([
            Block(768, 12, 4, qkv_bias=True,  norm_layer=nn.LayerNorm)
            for i in range(4)])
	
	def forward(self, s0, s1):
		s01 = torch.cat([s0, s1], dim=1)
		x=s01
		for block in self.blocks:
			x=block(x)
		x=x[:,:1,:]

		# x = self.conv(s01)
		# x = self.bn(x)
		# x = self.act(x)
		# for block in self.resblocks:
		# 	x = block(x)
		# x = self.conv_out(x)
		z, loss, perplexity, encodings = self.quantizer(x)
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
            Block(768, 12, 4, qkv_bias=True,  norm_layer=nn.LayerNorm)
            for i in range(4)])
	
	def forward(self, s, z):
		sz = torch.cat([s, z], dim=1)
		x=sz
		for block in self.blocks:
			x=block(x)
		x=x[:,:197*4,:]

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


if __name__ == '__main__':
	test_quantizer()
