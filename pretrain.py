import pickle
from pickletools import optimize
from pyexpat import model
from time import time
import matplotlib
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tools import momentum_update
from config import config
import numpy as np
from tools import log, log_setting, NT_Xent, renormalize, simsiam_distance, get_data_loader, get_eval_data_loader
import torch.nn.functional as F
from atari import AtariDataset
from ssv2 import ssv2Dataset,ssv2VideoDataset, ego4dDataset
from model import RepresentationNetwork
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tools import AddGaussianNoise
from model import Projector, Projector2, Decoder, LatentActionGen, Dynamic, conv3x3, VQVAE
from transform import Transforms
import os
import random
from test import prepare_model, show_image, show_latent_diff,save_preprocess
from util.pos_embed import get_2d_sincos_pos_embed
import lpips
from r3m import load_r3m
from sklearn.metrics.pairwise import cosine_similarity
import timm.optim.optim_factory as optim_factory
import math
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
import argparse
import cv2
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--opt', default='lamb', type=str, metavar='OPTIMIZER',
					help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
					help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
					help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
					help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
					help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
					help='weight decay (default: 0.05)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
					help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
					help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
					help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
					help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
					help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
					help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
					help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
					help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
					help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
					help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
					help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
					help='LR decay rate (default: 0.1)')


args = parser.parse_args()



s0_mean=0.002
s0_std=0.758
s1_mean=0.001
s1_std=0.752
diff_mean=-0.001 # s1-s0
diff_std=0.346


s_mean=torch.tensor([     0.020,      0.031,     -0.014,     -0.119,      0.012,      0.023,     -0.040,     -0.046,     -0.046,     -0.085,      0.041,      0.064,      0.020,      0.019,      0.021,     -0.029,      0.011,      0.005,     -0.004,      0.067,      0.033,      0.003,     -0.038,     -0.029,     -0.023,     -0.022,      0.049,     -0.018,     -0.025,      0.016,     -0.008,      0.005,      0.031,     -0.008,      0.034,     -0.006,      0.034,     -0.024,     -0.022,     -0.001,      0.016,      0.084,      0.016,     -0.008,     -1.407,      0.006,      0.057,     -0.062,     -0.006,      0.043,     -0.007,      0.019,     -0.001,     -0.024,     -0.001,     -0.130,      0.010,      0.027,      0.000,      0.066,      0.980,     -0.029,     -0.035,     -0.018,     -0.008,      0.040,     -0.017,     -0.021,      0.001,     -0.027,      0.002,     -0.004,      0.033,      0.030,     -0.003,     -0.030,     -0.033,     -0.001,     -0.436,     -0.059,      0.011,      0.087,     -0.002,      0.049,     -0.015,     -0.017,      0.060,     -0.029,      0.030,     -0.011,      0.026,      0.019,      0.134,     -0.014,      0.018,     -0.026,      0.039,      0.007,     -0.006,      0.548,     -0.000,      0.010,      0.006,      0.001,     -0.072,      0.020,      0.023,      0.024,      1.467,     -0.023,      0.014,     -0.018,      0.012,     -0.021,      0.012,      0.007,      0.049,      0.020,      0.001,     -0.004,      0.036,      0.011,      0.179,      0.016,     -0.010,      0.001,     -0.032,     -0.027,     -0.013,     -0.028,     -0.815,     -0.073,      0.005,      0.008,     -0.285,     -0.016,     -0.024,     -0.060,     -0.002,      0.015,     -0.028,      0.024,      0.033,      0.004,     -0.017,     -0.008,     -0.021,     -0.116,     -0.012,      0.009,      0.001,      0.019,     -0.002,     -0.002,      0.021,     -0.001,      0.023,     -0.006,      0.612,     -0.683,     -0.030,      0.012,     -0.027,     -0.012,     -0.258,     -0.020,      0.000,      0.008,     -0.329,      0.009,     -0.012,     -0.020,     -0.052,     -0.013,     -0.007,     -0.004,     -0.007,      0.766,      0.521,      0.001,      0.003,     -0.029,     -0.005,     -0.047,     -0.022,     -0.010,     -0.006,     -0.043,     -0.021,     -0.042,     -0.016,     -0.028,      0.029,      0.132,     -0.004,     -0.086,      0.023,     -0.024,      0.009,      0.014,      0.007,     -0.001,     -0.070,     -0.048,      0.028,      0.001,      0.013,      0.016,     -0.002,      0.010,      0.024,     -0.001,      0.008,      0.020,      0.020,     -0.011,      0.022,     -0.018,      0.006,      0.012,     -0.061,     -0.002,     -0.009,      0.008,      0.303,     -0.077,     -0.016,      0.053,      0.030,     -0.051,      0.103,     -0.009,     -0.067,     -0.003,      0.047,     -0.077,     -0.120,     -0.031,      0.027,     -0.077,      0.021,     -0.006,      0.000,      0.037,      0.046,      0.022,      0.032,      0.099,      0.018,     -0.012,     -0.008,      0.040,      0.011,     -0.034,      0.040,     -0.008,      0.020,     -0.024,     -0.015,     -0.027,      0.296,     -0.006,     -0.010,      0.085,      0.035,      0.022,      0.008,     -0.051,      0.027,      0.018,      0.006,      0.049,     -0.036,     -0.021,      0.050,     -0.026,      0.042,     -0.032,      0.027,     -0.007,     -0.036,      0.014,      0.014,      0.034,     -0.001,     -0.011,     -0.037,     -0.030,      0.005,     -0.012,     -0.037,     -0.028,      0.009,      0.063,     -0.025,     -0.024,      0.019,     -0.047,      0.005,     -0.065,     -0.000,      0.023,      0.049,     -0.051,     -0.004,      0.025,      0.000,     -0.009,     -0.046,      0.051,      0.024,      0.026,      0.022,     -0.028,     -0.003,     -0.059,      0.018,      0.007,      0.014,     -0.243,     -0.042,     -0.019,     -0.027,      0.012,      0.004,      0.002,      0.006,      0.014,     -0.085,     -0.021,      0.047,      0.038,     -0.062,      0.670,      0.016,     -0.041,     -0.019,     -0.047,     -0.004,      0.018,      0.016,     -0.012,     -0.015,     -0.007,      0.000,     -0.020,      0.011,     -0.045,     -0.007,     -0.003,     -0.029,     -0.002,     -0.055,      0.045,      0.027,     -0.077,      0.014,      0.055,     -0.278,      0.015,     -0.004,     -0.041,      0.006,     -0.028,     -0.020,      0.028,     -0.069,      0.004,      0.063,     -0.134,     -0.000,      0.081,     -0.137,     -0.001,     -0.033,      0.023,      0.021,      0.026,     -0.020,      0.001,      0.062,     -0.019,      0.077,     -0.051])
s_std=torch.tensor([0.121, 0.161, 0.057, 0.140, 0.096, 0.074, 0.238, 0.264, 0.138, 0.529, 0.203, 0.288, 0.106, 0.188, 0.090, 0.111, 0.125, 0.089, 0.120, 0.171, 0.236, 0.177, 0.187, 0.363, 0.226, 0.378, 0.349, 0.080, 0.110, 0.095, 0.116, 0.082, 0.122, 0.120, 0.164, 0.093, 0.203, 0.150, 0.105, 0.188, 0.116, 0.550, 0.068, 0.124, 1.485, 0.162, 0.143, 0.227, 0.145, 0.139, 0.111, 0.497, 0.151, 0.130, 0.111, 0.943, 0.072, 0.154, 0.091, 0.361, 0.307, 0.157, 0.088, 0.351, 0.178, 0.130, 0.400, 0.137, 0.233, 0.137, 0.187, 0.106, 0.139, 0.269, 0.155, 0.111, 0.089, 0.166, 0.743, 0.320, 0.321, 0.150, 0.122, 0.133, 0.146, 0.130, 0.123, 0.101, 0.163, 0.133, 0.108, 0.100, 0.529, 0.100, 0.122, 0.299, 0.107, 0.071, 0.063, 0.981, 0.101, 0.195, 0.392, 0.131, 0.249, 0.088, 0.261, 0.145, 2.113, 0.264, 0.119, 0.435, 0.231, 0.067, 0.173, 0.075, 0.137, 0.124, 0.109, 0.249, 0.252, 0.122, 0.736, 0.126, 0.102, 0.130, 0.117, 0.154, 0.141, 0.085, 1.691, 0.071, 0.150, 0.201, 0.751, 0.254, 0.111, 0.522, 0.195, 0.095, 0.411, 0.094, 0.143, 0.132, 0.132, 0.140, 0.132, 0.381, 0.107, 0.102, 0.103, 0.118, 0.075, 0.162, 0.080, 0.101, 0.144, 0.090, 0.838, 9.444, 0.141, 0.087, 0.105, 0.115, 0.670, 0.122, 0.125, 0.115, 0.700, 0.148, 0.120, 0.146, 0.088, 0.160, 0.114, 0.149, 0.123, 1.988, 0.765, 0.094, 0.122, 0.119, 0.079, 0.212, 0.135, 0.329, 0.055, 0.132, 0.105, 0.108, 0.088, 0.171, 0.163, 0.146, 0.082, 0.161, 0.129, 0.130, 0.102, 0.125, 0.097, 0.122, 0.243, 0.214, 0.205, 0.241, 0.153, 0.160, 0.190, 0.138, 0.113, 0.292, 0.110, 0.076, 0.164, 0.095, 0.104, 0.099, 0.149, 0.099, 0.160, 0.163, 0.104, 0.126, 0.453, 0.210, 0.130, 0.130, 0.176, 0.153, 0.523, 0.135, 0.377, 0.089, 0.202, 0.118, 0.129, 0.082, 0.121, 0.389, 0.152, 0.126, 0.143, 0.266, 0.187, 0.086, 0.146, 0.260, 0.096, 0.072, 0.170, 0.117, 0.171, 0.254, 0.110, 0.126, 0.193, 0.081, 0.125, 0.125, 0.642, 0.086, 0.114, 0.422, 0.098, 0.098, 0.093, 0.168, 0.230, 0.117, 0.179, 0.102, 0.102, 0.136, 0.128, 0.100, 0.196, 0.121, 0.129, 0.127, 0.162, 0.139, 0.117, 0.139, 0.136, 0.146, 0.099, 0.119, 0.079, 0.118, 0.166, 0.135, 0.161, 0.165, 0.083, 0.186, 0.245, 0.153, 0.120, 0.130, 0.230, 0.316, 0.323, 0.246, 0.147, 0.080, 0.122, 0.128, 0.115, 0.121, 0.277, 0.125, 0.089, 0.099, 0.135, 0.153, 0.147, 0.102, 0.073, 0.418, 0.117, 0.143, 0.106, 0.089, 0.229, 0.082, 0.131, 0.111, 0.297, 0.128, 0.147, 0.281, 0.386, 0.802, 0.146, 0.161, 0.117, 0.134, 0.459, 0.104, 0.110, 0.176, 0.073, 0.139, 0.223, 0.623, 0.150, 0.137, 0.132, 0.089, 0.175, 0.093, 0.122, 0.134, 0.133, 0.332, 0.107, 0.151, 0.515, 0.187, 0.094, 0.108, 0.082, 0.168, 0.211, 0.129, 0.280, 0.101, 0.156, 0.516, 0.066, 0.250, 0.184, 0.182, 0.188, 0.125, 0.098, 0.111, 0.189, 0.149, 0.218, 0.150, 0.176, 0.104])



# folder_name='./eval_visualization_RR_24_1loss_256_4_2_vits_metric'
# folder_name='./eval_visualization_RR_24_s_1024_42_bs256_adamW_cos_1e-3_standard'
folder_name='./robonet_test'
pyplot_cnt = 0

folder=os.path.exists(folder_name)
subfolder=os.path.exists(folder_name+'/metric_data_all')
if not folder:
	os.makedirs(folder_name)
if not subfolder:
	os.makedirs(folder_name+'/metric_data_all')

# set random seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



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


def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



class Model(nn.Module):
	def __init__(self, name='naive', num_channels=768, transform=None):
		super(Model, self).__init__()
		self.name = name
		# self.encoder = RepresentationNetwork(config.ss_observation_shape,
		#                                      num_blocks=5,
		#                                      num_channels=num_channels,
		#                                      downsample=True,
		#                                      momentum=config.bn_momentum)
		# self.decoder = Decoder()
		# self.loss_fn_vgg = lpips.LPIPS(net='vgg')
		# self.r3m = load_r3m("resnet50") # resnet18, resnet34
		# self.r3m.eval()
		# self.r3m.to(config.device)
		# print(self.r3m.module.device)
		# print(self.r3m.device_ids)
		# quit()
		# self.model_mae= prepare_model(chkpt_dir='./mae_visualize_vit_base.pth', arch='mae_vit_base_patch16',device=config.device)
		self.model_mae= prepare_model(chkpt_dir='./checkpoint-1599.pth', arch='mae_vit_small_patch16',device=config.device)
		self.latent_dim=config.latent_dim
		self.latent_num=24
		self.model_mae.requires_grad_(False)
		# freeze mae encoder parameters
		# self.model_mae.blocks.requires_grad_(False)
		# self.model_mae.decoder_blocks.requires_grad_(False)
		self.lag = LatentActionGen(config.num_embeddings,
		                           num_channels,
		                           config.latent_action_channel,
		                           num_blocks=5)
		self.dynamic = Dynamic(num_channels, config.latent_action_channel, num_blocks=5)
		# self.projector = Projector(num_channels, 10)
		# self.predictor = Predictor()
		
		self.num_channels = num_channels
		self.transform = transform
		
		# self.optim = optim.Optimizer(self.parameters(), {})
		# self.loss = nn.CosineSimilarity()
		
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
		# self.projection_in_dim = num_channels * config.state_size
		# self.projection = nn.Sequential(
		# 	nn.Linear(self.projection_in_dim, self.proj_hid, bias=False),
		# 	nn.BatchNorm1d(self.proj_hid),
		# 	nn.ReLU(),
		# 	nn.Linear(self.proj_hid, self.proj_hid, bias=False),
		# 	nn.BatchNorm1d(self.proj_hid),
		# 	nn.ReLU(),
		# 	nn.Linear(self.proj_hid, self.proj_out),
		# 	nn.BatchNorm1d(self.proj_out, affine=False)
		# )
		# self.projection_head = nn.Sequential(
		# 	nn.Linear(self.proj_out, self.pred_hid, bias=False),
		# 	nn.BatchNorm1d(self.pred_hid),
		# 	nn.ReLU(),
		# 	nn.Linear(self.pred_hid, self.pred_out),
		# )
		self.pos_embed_set = nn.Parameter(torch.zeros(1, 4*196 + 1, self.latent_dim), requires_grad=False)  # fixed sin-cos embedding
		self.produced_latent = nn.Parameter(torch.zeros(config.batch_size, 197, self.latent_dim))
		self.latent_diff = nn.Parameter(torch.zeros(config.batch_size, self.latent_num, self.latent_dim))
		self.initial_weight()

	def initial_weight(self):
		pos_embed_set = get_2d_sincos_pos_embed(self.latent_dim, int(28), cls_token=True)
		self.pos_embed_set.data.copy_(torch.from_numpy(pos_embed_set).float().unsqueeze(0))	
		torch.nn.init.normal_(self.produced_latent, std=.02)
		torch.nn.init.normal_(self.latent_diff, std=.02)	

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

	def visualize(self,obs0,_obs0,obs1,_obs1,obs1_blur,s1, _s1,cnt=0):
		# obs1=obs1_blur
		# _obs0_cal=obs0[:, -3:][0].unsqueeze(0).to(torch.float32)
		# __obs0_cal=_obs0[:, -3:][0].unsqueeze(0).to(torch.float32)
		# _obs1_cal=obs1[:, -3:][0].unsqueeze(0).to(torch.float32)
		# __obs1_cal=_obs1[:, -3:][0].unsqueeze(0).to(torch.float32)
		# d_0=self.loss_fn_alex(_obs1_cal,__obs0_cal).item()
		# d_1=self.loss_fn_alex(_obs1_cal,__obs1_cal).item()
		# d_0=round(d_0,3)
		# d_1=round(d_1,3)
		# print(_obs0_cal.device)
		# print(_obs0_cal.shape)

		# # r3m
		# with torch.no_grad():
		# 	embedding_obs0 = self.r3m(_obs0_cal * 255.0) ## R3M expects image input to be [0-255]
		# 	embedding_recon0 = self.r3m(__obs0_cal * 255.0)
		# 	embedding_obs1 = self.r3m(_obs1_cal * 255.0)
		# 	embedding_recon1 = self.r3m(__obs1_cal * 255.0)
		# # print(embedding.shape) # [1, 2048]
		# d_0_cos=cosine_similarity(embedding_obs1,embedding_recon0).item()
		# d_1_cos=cosine_similarity(embedding_obs1,embedding_recon1).item()
		# d_0_cos=round(d_0_cos,3)
		# d_1_cos=round(d_1_cos,3)


		# print(d_0,d_1)
		# print(_obs0_cal.shape,__obs0_cal.shape,_obs1_cal.shape,__obs1_cal.shape)
		# quit()
		# s0,mask0,ids_restore0=self.mae_encoder_forward(obs0)
		# _obs0=self.mae_decoder_forward(s0,ids_restore0)
		# _obs0=self.model_mae.unpatchify(_obs0)
		_obs0_ = torch.einsum('nchw->nhwc', _obs0).detach().cpu()
		obs0_ = torch.einsum('nchw->nhwc', obs0[:, -3:]).detach().cpu()
		_obs1_ = torch.einsum('nchw->nhwc', _obs1).detach().cpu()
		obs1_ = torch.einsum('nchw->nhwc', obs1[:, -3:]).detach().cpu()
		latent_mse=((s1[0]-_s1[0])**2).mean(dim=-1)
		latent_mse=latent_mse[1:].reshape(14,14).detach().cpu()
		# plt.rcParams['figure.figsize'] = [24, 24]
		plt.subplot(1, 5, 1)
		show_image(obs0_[0], "obs_0")
		plt.subplot(1, 5, 2)
		show_image(_obs0_[0], "PL:"+str('d_0')+"\nemb:"+str('d_0_cos'))
		plt.subplot(1, 5, 3)
		show_image(obs1_[0], "obs_1")
		plt.subplot(1, 5, 4)
		show_image(_obs1_[0], "PL:"+str('d_1')+"\nemb:"+str('d_1_cos'))
		plt.subplot(1,5,5)
		show_latent_diff(latent_mse=latent_mse, title="s_mse")
		plt.show()
		plt.savefig(folder_name+'/test_4_42_'+str(cnt)+'.png')
		# plt.savefig('./eval_visualization_48_42/test_4_42_'+str(cnt)+'.png')
		plt.close()



	def save_fig(self,obs0,_obs0,obs1,_obs1,obs1_blur,cnt=0):

		for i in range(config.batch_size):
		# if _obs0.device=='cuda:0':
			deviceid_r0=str(_obs0.device)
			_obs0_ = torch.einsum('nchw->nhwc', _obs0).detach().cpu()
			img_recon0=save_preprocess(_obs0_[i])
			plt.imsave(folder_name+'/metric_data_all/recon0_'+str(i)+'_'+str(cnt)+deviceid_r0+'.png',img_recon0)
			plt.close()
		# if obs0.device=='cuda:0':
			devicei_o0=str(obs0.device)
			obs0_ = torch.einsum('nchw->nhwc', obs0).detach().cpu()
			img_obs0=save_preprocess(obs0_[i])
			plt.imsave(folder_name+'/metric_data_all/obs0_'+str(i)+'_'+str(cnt)+devicei_o0+'.png',img_obs0)
			plt.close()
		# if _obs1.device=='cuda:0':
			devicei_r1=str(obs0.device)
			_obs1_ = torch.einsum('nchw->nhwc', _obs1).detach().cpu()
			img_recon1=save_preprocess(_obs1_[i])
			plt.imsave(folder_name+'/metric_data_all/recon1_'+str(i)+'_'+str(cnt)+devicei_r1+'.png',img_recon1)
			plt.close()
		# if obs1.device=='cuda:0':
			devicei_10=str(obs0.device)
			obs1_ = torch.einsum('nchw->nhwc', obs1).detach().cpu()
			img_obs1=save_preprocess(obs1_[i])
			plt.imsave(folder_name+'/metric_data_all/obs1_'+str(i)+'_'+str(cnt)+devicei_10+'.png',img_obs1)
			plt.close()
		# if obs1_blur.device=='cuda:0':
			devicei_b=str(obs0.device)
			obs1_blur_=torch.einsum('nchw->nhwc', obs1_blur).detach().cpu()
			img_obs1_blur=save_preprocess(obs1_blur_[i])
			plt.imsave(folder_name+'/metric_data_all/obs1_blur_'+str(i)+'_'+str(cnt)+devicei_b+'.png',img_obs1_blur)
			plt.close()
			# img_obs0=save_preprocess(obs0_[0])
			# img_recon0=save_preprocess(_obs0_[0])
			# img_obs1=save_preprocess(obs1_[0])
			# img_recon1=save_preprocess(_obs1_[0])
			# img_obs1_blur=save_preprocess(obs1_blur[0])
			# # plt.imshow(obs0_[0])
			# # print(obs0_[0].shape)
			# # quit()
			# plt.imsave('./metric_data/obs0_'+str(cnt)+'.png',img_obs0)
			# # plt.imshow(_obs0_[0])
			# plt.imsave('./metric_data/recon0_'+str(cnt)+'.png',img_recon0)
			# # plt.imshow(obs1_[0])
			# plt.imsave('./metric_data/obs1_'+str(cnt)+'.png',img_obs1)
			# # plt.imshow(_obs1_[0])
			# plt.imsave('./metric_data/recon1_'+str(cnt)+'.png',img_recon1)
			# # plt.imshow(obs1_blur[0])
			# plt.imsave('./metric_data/obs1_blur_'+str(cnt)+'.png',img_obs1_blur)
			plt.close()
			# plt.subplot(1, 5, 1)
			# show_image(obs0_[0], "obs_0")
			# plt.subplot(1, 5, 2)
			# show_image(_obs0_[0], ":"+str(d_0))
			# plt.subplot(1, 5, 3)
			# show_image(obs1_[0], "obs_1")
			# plt.subplot(1, 5, 4)
			# show_image(_obs1_[0], ":"+str(d_1))
			# plt.subplot(1,5,5)
			# show_latent_diff(latent_mse=latent_mse, title="s_mse")
			# plt.show()
			# plt.savefig('./metric_fig/'+str(cnt)+'.png')
			# # plt.savefig('./eval_visualization_48_42/test_4_42_'+str(cnt)+'.png')
			# plt.close()
		


	def forward(self,obs0,obs1,s_mean,s_std):

		# auto encoder loss
		s0,mask0,ids_restore0=self.mae_encoder_forward(obs0)
		s1,mask1,ids_restore1=self.mae_encoder_forward(obs1)
		# # if 4 frames stack
		# _obs0=self.mae_decoder_forward(s0[:,-197:,:],ids_restore0[:,-196:])
		# _obs0=self.model_mae.unpatchify(_obs0)

		# if single frame
		_obs0=self.mae_decoder_forward(s0,ids_restore0)
		_obs0=self.model_mae.unpatchify(_obs0)
		# trick blur obs_1
		obs1_blur=self.mae_decoder_forward(s1,ids_restore1)
		obs1_blur=self.model_mae.unpatchify(obs1_blur)


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

			# normalize s0 and s1
			s0_dummy=s0
			# s_mean=s_mean.to(config.device)
			# s_std=s_std.to(config.device)
			# s0=(s0-s_mean)/s_std
			# s1=(s1-s_mean)/s_std
			# print('mean and std:',s0.std())
			# quit()
			
			z, loss_lag, perp = self.lag(s0, s1, self.pos_embed_set, self.latent_diff)
			_s1 = self.dynamic(s0, z, self.pos_embed_set,self.produced_latent)
			s1_out=_s1
			# s1_out = self.dynamic(s0, z, self.pos_embed_set,self.produced_latent)
			# s1_out=s1_out*diff_std+diff_mean
			# s0=s0*s0_std+s0_mean
			# s1=s1*s1_std+s1_mean
			# _s1=s1_out+s0
			# _s1=_s1*s_std+s_mean
			# s0=s0*s_std+s_mean
			# s1=s1*s_std+s_mean



			# # if 4 frames stack
			# _obs1=self.mae_decoder_forward(_s1[:,-197:,:],ids_restore1[:,-196:])
			# _obs1=self.model_mae.unpatchify(_obs1)
			# if single frame
			_obs1=self.mae_decoder_forward(_s1,ids_restore1)
			_obs1=self.model_mae.unpatchify(_obs1)			
		return obs0, obs1, _obs0, _obs1,obs1_blur, s0, s1_out, s1, _s1, loss_lag

	def calculate_loss(self, obs0, obs1, _obs0, _obs1, s0, s1_out, s1, _s1, loss_lag):
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

		# s0=s0*s_std+s_mean
		# s1=s1*s_std+s_mean
		# s_diff=s1-s0
		# s_diff=(s_diff-diff_mean)/diff_std
		# print('mean and std:',s_diff.mean())
		# quit()
		# loss_dyna = F.mse_loss(s1_out,s_diff)
		loss_dyna = F.mse_loss(s1,_s1)


		# total loss
		loss =  loss_lag + loss_dyna # + loss_repr_dyn
		# loss=loss_lag
		# print('%.5f %.5f %.5f' % (loss_repr_dyn, loss_dyna, loss_lag))
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


def hook_f(grad):
	print(grad)


def training_curve(epoch_loss,mode):
	n=len(epoch_loss)
	index=range(n)
	if mode=='train':
		plt.plot(index,epoch_loss,label='train loss')
		plt.xlabel("iterations/10")
		plt.ylabel("loss")
		plt.ylim(0, 0.5)
		# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.legend()
		plt.title('freeze encoder and decoder')
		plt.savefig(folder_name+'/train loss_4_42.jpg')
		# plt.savefig('./eval_visualization_48_42/train loss_4_42.jpg')
		plt.close()
	elif mode=='eval':
		plt.plot(index,epoch_loss,label='eval loss')
		plt.xlabel("epoch")
		plt.ylabel("loss")
		plt.ylim(0, 0.5)
		# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
		plt.legend()
		plt.title('freeze encoder and decoder')
		plt.savefig(folder_name+'/evaluation loss_4_42.jpg')
		# plt.savefig('./eval_visualization_48_42/evaluation loss_4_42.jpg')
		plt.close()


def evaluation(model,eval_dataset,eval_sampler,iter,epoch):
	cnt=0
	model.eval()
	eval_loss_sum=0
	eval_data_loader=get_eval_data_loader(eval_dataset,eval_sampler)
	for data in eval_data_loader:
		cnt+=1
		obs0, obs1 =data
		obs0 = obs0.type(torch.float32).to(config.device) / 255
		obs1 = obs1.type(torch.float32).to(config.device) / 255
		# normalize
		imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
		imagenet_std = torch.tensor([0.229, 0.224, 0.225])
		obs0 = torch.einsum('nchw->nhwc', obs0)
		obs1 = torch.einsum('nchw->nhwc', obs1)
		imagenet_mean=imagenet_mean.to(config.device)
		imagenet_std=imagenet_std.to(config.device)
		obs0=obs0-imagenet_mean
		obs0=obs0/imagenet_std
		obs1=obs1-imagenet_mean
		obs1=obs1/imagenet_std
		obs0 = torch.einsum('nhwc->nchw', obs0)
		obs1 = torch.einsum('nhwc->nchw', obs1)
		
		s_mean=torch.tensor([     0.020,      0.031,     -0.014,     -0.119,      0.012,      0.023,     -0.040,     -0.046,     -0.046,     -0.085,      0.041,      0.064,      0.020,      0.019,      0.021,     -0.029,      0.011,      0.005,     -0.004,      0.067,      0.033,      0.003,     -0.038,     -0.029,     -0.023,     -0.022,      0.049,     -0.018,     -0.025,      0.016,     -0.008,      0.005,      0.031,     -0.008,      0.034,     -0.006,      0.034,     -0.024,     -0.022,     -0.001,      0.016,      0.084,      0.016,     -0.008,     -1.407,      0.006,      0.057,     -0.062,     -0.006,      0.043,     -0.007,      0.019,     -0.001,     -0.024,     -0.001,     -0.130,      0.010,      0.027,      0.000,      0.066,      0.980,     -0.029,     -0.035,     -0.018,     -0.008,      0.040,     -0.017,     -0.021,      0.001,     -0.027,      0.002,     -0.004,      0.033,      0.030,     -0.003,     -0.030,     -0.033,     -0.001,     -0.436,     -0.059,      0.011,      0.087,     -0.002,      0.049,     -0.015,     -0.017,      0.060,     -0.029,      0.030,     -0.011,      0.026,      0.019,      0.134,     -0.014,      0.018,     -0.026,      0.039,      0.007,     -0.006,      0.548,     -0.000,      0.010,      0.006,      0.001,     -0.072,      0.020,      0.023,      0.024,      1.467,     -0.023,      0.014,     -0.018,      0.012,     -0.021,      0.012,      0.007,      0.049,      0.020,      0.001,     -0.004,      0.036,      0.011,      0.179,      0.016,     -0.010,      0.001,     -0.032,     -0.027,     -0.013,     -0.028,     -0.815,     -0.073,      0.005,      0.008,     -0.285,     -0.016,     -0.024,     -0.060,     -0.002,      0.015,     -0.028,      0.024,      0.033,      0.004,     -0.017,     -0.008,     -0.021,     -0.116,     -0.012,      0.009,      0.001,      0.019,     -0.002,     -0.002,      0.021,     -0.001,      0.023,     -0.006,      0.612,     -0.683,     -0.030,      0.012,     -0.027,     -0.012,     -0.258,     -0.020,      0.000,      0.008,     -0.329,      0.009,     -0.012,     -0.020,     -0.052,     -0.013,     -0.007,     -0.004,     -0.007,      0.766,      0.521,      0.001,      0.003,     -0.029,     -0.005,     -0.047,     -0.022,     -0.010,     -0.006,     -0.043,     -0.021,     -0.042,     -0.016,     -0.028,      0.029,      0.132,     -0.004,     -0.086,      0.023,     -0.024,      0.009,      0.014,      0.007,     -0.001,     -0.070,     -0.048,      0.028,      0.001,      0.013,      0.016,     -0.002,      0.010,      0.024,     -0.001,      0.008,      0.020,      0.020,     -0.011,      0.022,     -0.018,      0.006,      0.012,     -0.061,     -0.002,     -0.009,      0.008,      0.303,     -0.077,     -0.016,      0.053,      0.030,     -0.051,      0.103,     -0.009,     -0.067,     -0.003,      0.047,     -0.077,     -0.120,     -0.031,      0.027,     -0.077,      0.021,     -0.006,      0.000,      0.037,      0.046,      0.022,      0.032,      0.099,      0.018,     -0.012,     -0.008,      0.040,      0.011,     -0.034,      0.040,     -0.008,      0.020,     -0.024,     -0.015,     -0.027,      0.296,     -0.006,     -0.010,      0.085,      0.035,      0.022,      0.008,     -0.051,      0.027,      0.018,      0.006,      0.049,     -0.036,     -0.021,      0.050,     -0.026,      0.042,     -0.032,      0.027,     -0.007,     -0.036,      0.014,      0.014,      0.034,     -0.001,     -0.011,     -0.037,     -0.030,      0.005,     -0.012,     -0.037,     -0.028,      0.009,      0.063,     -0.025,     -0.024,      0.019,     -0.047,      0.005,     -0.065,     -0.000,      0.023,      0.049,     -0.051,     -0.004,      0.025,      0.000,     -0.009,     -0.046,      0.051,      0.024,      0.026,      0.022,     -0.028,     -0.003,     -0.059,      0.018,      0.007,      0.014,     -0.243,     -0.042,     -0.019,     -0.027,      0.012,      0.004,      0.002,      0.006,      0.014,     -0.085,     -0.021,      0.047,      0.038,     -0.062,      0.670,      0.016,     -0.041,     -0.019,     -0.047,     -0.004,      0.018,      0.016,     -0.012,     -0.015,     -0.007,      0.000,     -0.020,      0.011,     -0.045,     -0.007,     -0.003,     -0.029,     -0.002,     -0.055,      0.045,      0.027,     -0.077,      0.014,      0.055,     -0.278,      0.015,     -0.004,     -0.041,      0.006,     -0.028,     -0.020,      0.028,     -0.069,      0.004,      0.063,     -0.134,     -0.000,      0.081,     -0.137,     -0.001,     -0.033,      0.023,      0.021,      0.026,     -0.020,      0.001,      0.062,     -0.019,      0.077,     -0.051])
		s_std=torch.tensor([0.121, 0.161, 0.057, 0.140, 0.096, 0.074, 0.238, 0.264, 0.138, 0.529, 0.203, 0.288, 0.106, 0.188, 0.090, 0.111, 0.125, 0.089, 0.120, 0.171, 0.236, 0.177, 0.187, 0.363, 0.226, 0.378, 0.349, 0.080, 0.110, 0.095, 0.116, 0.082, 0.122, 0.120, 0.164, 0.093, 0.203, 0.150, 0.105, 0.188, 0.116, 0.550, 0.068, 0.124, 1.485, 0.162, 0.143, 0.227, 0.145, 0.139, 0.111, 0.497, 0.151, 0.130, 0.111, 0.943, 0.072, 0.154, 0.091, 0.361, 0.307, 0.157, 0.088, 0.351, 0.178, 0.130, 0.400, 0.137, 0.233, 0.137, 0.187, 0.106, 0.139, 0.269, 0.155, 0.111, 0.089, 0.166, 0.743, 0.320, 0.321, 0.150, 0.122, 0.133, 0.146, 0.130, 0.123, 0.101, 0.163, 0.133, 0.108, 0.100, 0.529, 0.100, 0.122, 0.299, 0.107, 0.071, 0.063, 0.981, 0.101, 0.195, 0.392, 0.131, 0.249, 0.088, 0.261, 0.145, 2.113, 0.264, 0.119, 0.435, 0.231, 0.067, 0.173, 0.075, 0.137, 0.124, 0.109, 0.249, 0.252, 0.122, 0.736, 0.126, 0.102, 0.130, 0.117, 0.154, 0.141, 0.085, 1.691, 0.071, 0.150, 0.201, 0.751, 0.254, 0.111, 0.522, 0.195, 0.095, 0.411, 0.094, 0.143, 0.132, 0.132, 0.140, 0.132, 0.381, 0.107, 0.102, 0.103, 0.118, 0.075, 0.162, 0.080, 0.101, 0.144, 0.090, 0.838, 9.444, 0.141, 0.087, 0.105, 0.115, 0.670, 0.122, 0.125, 0.115, 0.700, 0.148, 0.120, 0.146, 0.088, 0.160, 0.114, 0.149, 0.123, 1.988, 0.765, 0.094, 0.122, 0.119, 0.079, 0.212, 0.135, 0.329, 0.055, 0.132, 0.105, 0.108, 0.088, 0.171, 0.163, 0.146, 0.082, 0.161, 0.129, 0.130, 0.102, 0.125, 0.097, 0.122, 0.243, 0.214, 0.205, 0.241, 0.153, 0.160, 0.190, 0.138, 0.113, 0.292, 0.110, 0.076, 0.164, 0.095, 0.104, 0.099, 0.149, 0.099, 0.160, 0.163, 0.104, 0.126, 0.453, 0.210, 0.130, 0.130, 0.176, 0.153, 0.523, 0.135, 0.377, 0.089, 0.202, 0.118, 0.129, 0.082, 0.121, 0.389, 0.152, 0.126, 0.143, 0.266, 0.187, 0.086, 0.146, 0.260, 0.096, 0.072, 0.170, 0.117, 0.171, 0.254, 0.110, 0.126, 0.193, 0.081, 0.125, 0.125, 0.642, 0.086, 0.114, 0.422, 0.098, 0.098, 0.093, 0.168, 0.230, 0.117, 0.179, 0.102, 0.102, 0.136, 0.128, 0.100, 0.196, 0.121, 0.129, 0.127, 0.162, 0.139, 0.117, 0.139, 0.136, 0.146, 0.099, 0.119, 0.079, 0.118, 0.166, 0.135, 0.161, 0.165, 0.083, 0.186, 0.245, 0.153, 0.120, 0.130, 0.230, 0.316, 0.323, 0.246, 0.147, 0.080, 0.122, 0.128, 0.115, 0.121, 0.277, 0.125, 0.089, 0.099, 0.135, 0.153, 0.147, 0.102, 0.073, 0.418, 0.117, 0.143, 0.106, 0.089, 0.229, 0.082, 0.131, 0.111, 0.297, 0.128, 0.147, 0.281, 0.386, 0.802, 0.146, 0.161, 0.117, 0.134, 0.459, 0.104, 0.110, 0.176, 0.073, 0.139, 0.223, 0.623, 0.150, 0.137, 0.132, 0.089, 0.175, 0.093, 0.122, 0.134, 0.133, 0.332, 0.107, 0.151, 0.515, 0.187, 0.094, 0.108, 0.082, 0.168, 0.211, 0.129, 0.280, 0.101, 0.156, 0.516, 0.066, 0.250, 0.184, 0.182, 0.188, 0.125, 0.098, 0.111, 0.189, 0.149, 0.218, 0.150, 0.176, 0.104])
		s_mean=s_mean.to(config.device)
		s_std=s_std.to(config.device)
		with torch.no_grad():
			obs0, obs1, _obs0, _obs1,obs1_blur,s0, s1_out, s1, _s1, loss_lag=model(obs0, obs1,s_mean,s_std)
			loss=model.module.calculate_loss(obs0, obs1, _obs0, _obs1,s0, s1_out, s1, _s1, loss_lag)
		eval_loss_sum+=loss.mean().item()
		if True:
			model.module.visualize(obs0,_obs0,obs1,_obs1,obs1_blur,s1, _s1,cnt)
		if epoch%10==9:
			model.module.save_fig(obs0,_obs0,obs1,_obs1,obs1_blur,cnt)
	return eval_loss_sum/cnt



def train_epoch(model, dataset, optimizer,sampler,eval_dataset,eval_sampler):
	# lr_scheduler, _ = create_scheduler(args, optimizer)
	loss_curve=[]
	eval_loss_curve=[]
	cnt = 0
	# s0_list=0
	# s1_list=0
	# diff_list=[]
	# s0_std_list=0
	# s1_std_list=0
	# diff_std_list=[]
	for epoch in range(config.epochs):
		data_iter_step=0
		data_loader = get_data_loader(dataset,sampler)
		for data in data_loader:
			# we use a per iteration (instead of per epoch) lr scheduler
			if data_iter_step % 1 == 0:
				adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch)
			model.train()
			print(cnt)
			# print(data.shape)
			# (obs0, obs1), action, reward = data
			obs0, obs1 =data
			obs0 = obs0.type(torch.float32).to(config.device) / 255
			obs1 = obs1.type(torch.float32).to(config.device) / 255


			# normalize
			imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
			imagenet_std = torch.tensor([0.229, 0.224, 0.225])
			obs0 = torch.einsum('nchw->nhwc', obs0)
			obs1 = torch.einsum('nchw->nhwc', obs1)
			imagenet_mean=imagenet_mean.to(config.device)
			imagenet_std=imagenet_std.to(config.device)
			obs0=obs0-imagenet_mean
			obs0=obs0/imagenet_std
			obs1=obs1-imagenet_mean
			obs1=obs1/imagenet_std
			obs0 = torch.einsum('nhwc->nchw', obs0)
			obs1 = torch.einsum('nhwc->nchw', obs1)
			
			s_mean=torch.tensor([     0.020,      0.031,     -0.014,     -0.119,      0.012,      0.023,     -0.040,     -0.046,     -0.046,     -0.085,      0.041,      0.064,      0.020,      0.019,      0.021,     -0.029,      0.011,      0.005,     -0.004,      0.067,      0.033,      0.003,     -0.038,     -0.029,     -0.023,     -0.022,      0.049,     -0.018,     -0.025,      0.016,     -0.008,      0.005,      0.031,     -0.008,      0.034,     -0.006,      0.034,     -0.024,     -0.022,     -0.001,      0.016,      0.084,      0.016,     -0.008,     -1.407,      0.006,      0.057,     -0.062,     -0.006,      0.043,     -0.007,      0.019,     -0.001,     -0.024,     -0.001,     -0.130,      0.010,      0.027,      0.000,      0.066,      0.980,     -0.029,     -0.035,     -0.018,     -0.008,      0.040,     -0.017,     -0.021,      0.001,     -0.027,      0.002,     -0.004,      0.033,      0.030,     -0.003,     -0.030,     -0.033,     -0.001,     -0.436,     -0.059,      0.011,      0.087,     -0.002,      0.049,     -0.015,     -0.017,      0.060,     -0.029,      0.030,     -0.011,      0.026,      0.019,      0.134,     -0.014,      0.018,     -0.026,      0.039,      0.007,     -0.006,      0.548,     -0.000,      0.010,      0.006,      0.001,     -0.072,      0.020,      0.023,      0.024,      1.467,     -0.023,      0.014,     -0.018,      0.012,     -0.021,      0.012,      0.007,      0.049,      0.020,      0.001,     -0.004,      0.036,      0.011,      0.179,      0.016,     -0.010,      0.001,     -0.032,     -0.027,     -0.013,     -0.028,     -0.815,     -0.073,      0.005,      0.008,     -0.285,     -0.016,     -0.024,     -0.060,     -0.002,      0.015,     -0.028,      0.024,      0.033,      0.004,     -0.017,     -0.008,     -0.021,     -0.116,     -0.012,      0.009,      0.001,      0.019,     -0.002,     -0.002,      0.021,     -0.001,      0.023,     -0.006,      0.612,     -0.683,     -0.030,      0.012,     -0.027,     -0.012,     -0.258,     -0.020,      0.000,      0.008,     -0.329,      0.009,     -0.012,     -0.020,     -0.052,     -0.013,     -0.007,     -0.004,     -0.007,      0.766,      0.521,      0.001,      0.003,     -0.029,     -0.005,     -0.047,     -0.022,     -0.010,     -0.006,     -0.043,     -0.021,     -0.042,     -0.016,     -0.028,      0.029,      0.132,     -0.004,     -0.086,      0.023,     -0.024,      0.009,      0.014,      0.007,     -0.001,     -0.070,     -0.048,      0.028,      0.001,      0.013,      0.016,     -0.002,      0.010,      0.024,     -0.001,      0.008,      0.020,      0.020,     -0.011,      0.022,     -0.018,      0.006,      0.012,     -0.061,     -0.002,     -0.009,      0.008,      0.303,     -0.077,     -0.016,      0.053,      0.030,     -0.051,      0.103,     -0.009,     -0.067,     -0.003,      0.047,     -0.077,     -0.120,     -0.031,      0.027,     -0.077,      0.021,     -0.006,      0.000,      0.037,      0.046,      0.022,      0.032,      0.099,      0.018,     -0.012,     -0.008,      0.040,      0.011,     -0.034,      0.040,     -0.008,      0.020,     -0.024,     -0.015,     -0.027,      0.296,     -0.006,     -0.010,      0.085,      0.035,      0.022,      0.008,     -0.051,      0.027,      0.018,      0.006,      0.049,     -0.036,     -0.021,      0.050,     -0.026,      0.042,     -0.032,      0.027,     -0.007,     -0.036,      0.014,      0.014,      0.034,     -0.001,     -0.011,     -0.037,     -0.030,      0.005,     -0.012,     -0.037,     -0.028,      0.009,      0.063,     -0.025,     -0.024,      0.019,     -0.047,      0.005,     -0.065,     -0.000,      0.023,      0.049,     -0.051,     -0.004,      0.025,      0.000,     -0.009,     -0.046,      0.051,      0.024,      0.026,      0.022,     -0.028,     -0.003,     -0.059,      0.018,      0.007,      0.014,     -0.243,     -0.042,     -0.019,     -0.027,      0.012,      0.004,      0.002,      0.006,      0.014,     -0.085,     -0.021,      0.047,      0.038,     -0.062,      0.670,      0.016,     -0.041,     -0.019,     -0.047,     -0.004,      0.018,      0.016,     -0.012,     -0.015,     -0.007,      0.000,     -0.020,      0.011,     -0.045,     -0.007,     -0.003,     -0.029,     -0.002,     -0.055,      0.045,      0.027,     -0.077,      0.014,      0.055,     -0.278,      0.015,     -0.004,     -0.041,      0.006,     -0.028,     -0.020,      0.028,     -0.069,      0.004,      0.063,     -0.134,     -0.000,      0.081,     -0.137,     -0.001,     -0.033,      0.023,      0.021,      0.026,     -0.020,      0.001,      0.062,     -0.019,      0.077,     -0.051])
			s_std=torch.tensor([0.121, 0.161, 0.057, 0.140, 0.096, 0.074, 0.238, 0.264, 0.138, 0.529, 0.203, 0.288, 0.106, 0.188, 0.090, 0.111, 0.125, 0.089, 0.120, 0.171, 0.236, 0.177, 0.187, 0.363, 0.226, 0.378, 0.349, 0.080, 0.110, 0.095, 0.116, 0.082, 0.122, 0.120, 0.164, 0.093, 0.203, 0.150, 0.105, 0.188, 0.116, 0.550, 0.068, 0.124, 1.485, 0.162, 0.143, 0.227, 0.145, 0.139, 0.111, 0.497, 0.151, 0.130, 0.111, 0.943, 0.072, 0.154, 0.091, 0.361, 0.307, 0.157, 0.088, 0.351, 0.178, 0.130, 0.400, 0.137, 0.233, 0.137, 0.187, 0.106, 0.139, 0.269, 0.155, 0.111, 0.089, 0.166, 0.743, 0.320, 0.321, 0.150, 0.122, 0.133, 0.146, 0.130, 0.123, 0.101, 0.163, 0.133, 0.108, 0.100, 0.529, 0.100, 0.122, 0.299, 0.107, 0.071, 0.063, 0.981, 0.101, 0.195, 0.392, 0.131, 0.249, 0.088, 0.261, 0.145, 2.113, 0.264, 0.119, 0.435, 0.231, 0.067, 0.173, 0.075, 0.137, 0.124, 0.109, 0.249, 0.252, 0.122, 0.736, 0.126, 0.102, 0.130, 0.117, 0.154, 0.141, 0.085, 1.691, 0.071, 0.150, 0.201, 0.751, 0.254, 0.111, 0.522, 0.195, 0.095, 0.411, 0.094, 0.143, 0.132, 0.132, 0.140, 0.132, 0.381, 0.107, 0.102, 0.103, 0.118, 0.075, 0.162, 0.080, 0.101, 0.144, 0.090, 0.838, 9.444, 0.141, 0.087, 0.105, 0.115, 0.670, 0.122, 0.125, 0.115, 0.700, 0.148, 0.120, 0.146, 0.088, 0.160, 0.114, 0.149, 0.123, 1.988, 0.765, 0.094, 0.122, 0.119, 0.079, 0.212, 0.135, 0.329, 0.055, 0.132, 0.105, 0.108, 0.088, 0.171, 0.163, 0.146, 0.082, 0.161, 0.129, 0.130, 0.102, 0.125, 0.097, 0.122, 0.243, 0.214, 0.205, 0.241, 0.153, 0.160, 0.190, 0.138, 0.113, 0.292, 0.110, 0.076, 0.164, 0.095, 0.104, 0.099, 0.149, 0.099, 0.160, 0.163, 0.104, 0.126, 0.453, 0.210, 0.130, 0.130, 0.176, 0.153, 0.523, 0.135, 0.377, 0.089, 0.202, 0.118, 0.129, 0.082, 0.121, 0.389, 0.152, 0.126, 0.143, 0.266, 0.187, 0.086, 0.146, 0.260, 0.096, 0.072, 0.170, 0.117, 0.171, 0.254, 0.110, 0.126, 0.193, 0.081, 0.125, 0.125, 0.642, 0.086, 0.114, 0.422, 0.098, 0.098, 0.093, 0.168, 0.230, 0.117, 0.179, 0.102, 0.102, 0.136, 0.128, 0.100, 0.196, 0.121, 0.129, 0.127, 0.162, 0.139, 0.117, 0.139, 0.136, 0.146, 0.099, 0.119, 0.079, 0.118, 0.166, 0.135, 0.161, 0.165, 0.083, 0.186, 0.245, 0.153, 0.120, 0.130, 0.230, 0.316, 0.323, 0.246, 0.147, 0.080, 0.122, 0.128, 0.115, 0.121, 0.277, 0.125, 0.089, 0.099, 0.135, 0.153, 0.147, 0.102, 0.073, 0.418, 0.117, 0.143, 0.106, 0.089, 0.229, 0.082, 0.131, 0.111, 0.297, 0.128, 0.147, 0.281, 0.386, 0.802, 0.146, 0.161, 0.117, 0.134, 0.459, 0.104, 0.110, 0.176, 0.073, 0.139, 0.223, 0.623, 0.150, 0.137, 0.132, 0.089, 0.175, 0.093, 0.122, 0.134, 0.133, 0.332, 0.107, 0.151, 0.515, 0.187, 0.094, 0.108, 0.082, 0.168, 0.211, 0.129, 0.280, 0.101, 0.156, 0.516, 0.066, 0.250, 0.184, 0.182, 0.188, 0.125, 0.098, 0.111, 0.189, 0.149, 0.218, 0.150, 0.176, 0.104])
			s_mean=s_mean.to(config.device)
			s_std=s_std.to(config.device)

			
			# model. optim.zero_grad()
			optimizer.zero_grad()
			


			# loss = model.learn(obs0, obs1, visual=(cnt % 50 == 0))
			# print('#', loss)
			obs0, obs1, _obs0, _obs1,obs1_blur,s0, s1_out, s1, _s1, loss_lag=model(obs0, obs1,s_mean,s_std)
			# normalize 
			# s0_mean=s0.mean(dim=(0,1))
			# s0_list+=s0_mean
			# # print(s0_list.shape)
			# s0_std=s0.std(dim=(1))
			# s0_std=s0_std.mean(dim=0)
			# s0_std_list+=s0_std
			# print(s0_std_list.shape)
			# s1_mean=s1.mean(dim=(1,2))
			# s1_list.extend(s1_mean)
			# s1_std=s1.std(dim=(1,2))
			# s1_std_list.extend(s1_std)
			# diff_mean=(s1-s0).mean(dim=(1,2))
			# diff_list.extend(diff_mean)
			# diff_std=(s1-s0).std(dim=(1,2))
			# diff_std_list.extend(diff_std)			
			loss=model.module.calculate_loss(obs0, obs1, _obs0, _obs1,s0, s1_out, s1, _s1, loss_lag)
			# model.module.latent_diff.register_hook(hook_f)
			# model.module.produced_latent.register_hook(hook_f)
			# model.module.lag.parameters().register_hook(hook_f)
			loss.mean().backward()
			# for p in [
			# 	model.module.lag.parameters(),
			# 	model.module.model_mae.decoder_blocks.parameters(),
			# 	model.module.latent_diff,
			# 	model.module.produced_latent,
			# 	model.module.model_mae.blocks.parameters(),
			# 	# model.module.model_mae.decoder_embed.parameters(),
			# 	# model.module.model_mae.decoder_norm.parameters(),
			# 	# model.module.model_mae.decoder_pred.parameters(),
			# 	model.module.dynamic.parameters()]:
			# 	total_norm = nn.utils.clip_grad_norm_(p, max_norm=5.0)
			# 	print('grad_norm:', total_norm)
			# for parms in model.module.latent_diff: 
			# 	print('-->name:', 'name', '-->grad_requirs:',parms.requires_grad, \
			# 	' -->grad_value:',parms.grad, 'if leaf node:',parms.is_leaf)
			optimizer.step()
			if cnt % 10 ==0:
				loss_curve.append(loss.mean().item())
				training_curve(loss_curve,mode='train')
			if cnt % 20 ==0:
				# model.module.visualize(obs0,_obs0,obs1,_obs1)
				a=1
				# model.module.visualize_embedding(obs0, obs1)
				# break
			if cnt % 30 ==0:
				print('start evaluation')
				eval_loss=evaluation(model,eval_dataset=eval_dataset,eval_sampler=eval_sampler,iter=cnt,epoch=epoch)
				eval_loss_curve.append(eval_loss)
				training_curve(eval_loss_curve,mode='eval')
			cnt += 1
			data_iter_step+=1
			# lr = optimizer.param_groups[0]["lr"]
			# print('lr:',lr)
			print('##', loss.mean().item())
		# lr_scheduler.step(epoch)
		# if cnt==200:
		# N=cnt
		# s0_list=torch.Tensor(s0_list)
		# s1_list=torch.Tensor(s1_list)
		# diff_list=torch.Tensor(diff_list)
		# s0_std_list=torch.Tensor(s0_std_list)
		# s1_std_list=torch.Tensor(s1_std_list)
		# diff_std_list=torch.Tensor(diff_std_list)
		# s0_list=s0_list/N
		# s0_std_list=s0_std_list/N
		# print('mean of s0:',s0_list)
		# print('std of s0:',s0_std_list)
		# print('len n:',N)
		# print('mean and std of s1:',s1_list.mean(),s1_std_list.mean())
		# print('mean and std of diff:',diff_list.mean(),diff_std_list.mean())
		# print('len of list:',N)
		# quit()
	print('final eval loss:',eval_loss)



def vqvae_recons(origin,recon,latent_recon):
	origin = torch.einsum('nchw->nhwc', origin).detach().cpu()
	recon = torch.einsum('nchw->nhwc', recon).detach().cpu()
	latent_recon = torch.einsum('nchw->nhwc', latent_recon).detach().cpu()	
	plt.rcParams['figure.figsize'] = [24, 24]
	plt.subplot(1, 3, 1)
	show_image(origin[0], "origin")
	plt.subplot(1, 3, 2)
	show_image(recon[0], "quantize recon")
	plt.subplot(1, 3, 3)
	show_image(latent_recon[0], "latent recon")
	plt.savefig('vqvae_recon.png')

def vqvae_train_epoch(model, dataset, optimizer,sampler):
	model.train()
	data_loader = get_data_loader(dataset,sampler)
	cnt = 0
	for data in data_loader:
		print(cnt)
		obs0, obs1 =data
		obs0 = obs0.type(torch.float32).to(config.device) / 255
		obs1 = obs1.type(torch.float32).to(config.device) / 255	
		optimizer.zero_grad()
		cnt += 1
		results,latent_recon=model(obs0)
		loss=model.module.loss_function(*results)
		loss.backward()
		for p in [
			model.module.encoder.parameters(),
			model.module.decoder.parameters(),
			model.module.vq_layer.parameters()]:
			total_norm = nn.utils.clip_grad_norm_(p, max_norm=1.0)
			# print('grad_norm:', total_norm)
		optimizer.step()
		if cnt % 50 ==0:
			vqvae_recons(results[1],results[0],latent_recon)
		print('##', loss.mean().item())


def get_train_dataset(subdir, block_id):
	dataset = AtariDataset(subdir, block_id)
	return dataset


def get_tune_dataset():
	dataset = AtariDataset(5, 30, 20000)
	return dataset


def pretrain():
	# folder=os.path.exists(folder_name)
	# subfolder=os.path.exists(folder_name+'/metric_data')
	# if not folder:
	# 	os.makedirs(folder_name)
	# if not subfolder:
	# 	os.makedirs(folder_name+'/metric_data')


	# setup random seed
	setup_seed(666)

	# distributed training initialize
	torch.distributed.init_process_group(backend="nccl")
	local_rank=torch.distributed.get_rank()
	torch.cuda.set_device(local_rank)
	config.device= torch.device("cuda",local_rank)


	row_image_transform = transforms.Compose([
		transforms.RandomCrop(224,pad_if_needed=True)
	])
	eval_image_transform = transforms.Compose([
		transforms.CenterCrop(224)
	])	
	transform = Transforms()
	model = Model('ssae', transform=transform)
	# model=VQVAE(in_channels=3,embedding_dim=64,num_embeddings=512)
	# for name,parameters in model.model_mae.named_parameters():
	# 	print(name)
	# # print(model)
	# quit()


	model.to(config.device)
	model=nn.SyncBatchNorm.convert_sync_batchnorm(model)
	eff_batch_size=config.batch_size*8
	if config.lr is None:  # only base_lr is specified
		config.lr = config.blr * eff_batch_size / 256	



	model=torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=True, find_unused_parameters=True)
	model_without_ddp = model.module
	param_groups = optim_factory.add_weight_decay(model_without_ddp, 0.05) # origin 0.05
	# broadcast_buffers=False ???????
	# model.module.set_optimizer()
	# set_optimizer()
	# optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
	# optimizer = optim.Adam(model.parameters(), lr=0.0003)
	optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
	# optimizer = create_optimizer(args, model_without_ddp)
	# r_scheduler, _ = create_scheduler(args, optimizer)
	# model.restore()
	# model.module.save()
	# exit(0)
	
	# log.set_model(model.module.name)
	# log_setting()

	
	# tune_dataset = get_tune_dataset()
	# # state_reconstruct_test(model, tune_dataset)
	# action_regress_test(model, tune_dataset)
	# del tune_dataset
	# config.freeze_encoder_stat = True
	
	cnt = 0
	subdir, block_id = 1, 25
	# lr_schedule = [0.0001, 0.001, 0.01, 0.0333, 0.0666, 0.1, 0.2, 0.4, 0.8, 1.0]

	# train_dataset = ssv2Dataset(image_path='/home/chc/dataset/ssv2_extracted_frames_5',transform=row_image_transform,cut=None,mode='train')
	# eval_dataset = ssv2Dataset(image_path='/home/chc/dataset/ssv2_extracted_frames_5',transform=eval_image_transform,cut=None,mode='eval')
	# train_dataset = ssv2Dataset(image_path='/public/share_dataset/ssv2_extracted_frames_5',transform=row_image_transform,cut=None,mode='train')
	# eval_dataset = ssv2Dataset(image_path='/public/share_dataset/ssv2_extracted_frames_5',transform=eval_image_transform,cut=None,mode='eval')
	# use cache
	# train_dataset = ssv2Dataset(image_path='/cache0/cuihanchen/ssv2_extracted_frames_5',transform=row_image_transform,cut=None,mode='train')
	# eval_dataset = ssv2Dataset(image_path='/cache0/cuihanchen/ssv2_extracted_frames_5',transform=eval_image_transform,cut=None,mode='eval')
	# read from video
	# train_dataset = ssv2VideoDataset(image_path='/public/MARS/datasets/ssv2/20bn-something-something-v2',transform=row_image_transform,cut=None,mode='train')
	# eval_dataset = ssv2VideoDataset(image_path='/public/MARS/datasets/ssv2/20bn-something-something-v2',transform=row_image_transform,cut=None,mode='eval')	
	# ego4d toy
	# train_dataset = ego4dDataset(image_path='/cache0/cuihanchen/frames',transform=row_image_transform,cut=None,mode='train')
	# eval_dataset = ego4dDataset(image_path='/cache0/cuihanchen/frames',transform=row_image_transform,cut=None,mode='eval')	
	train_dataset = ego4dDataset(image_path='/public/share_dataset/chc/robonet_frames',transform=row_image_transform,cut=None,mode='train')
	eval_dataset = ego4dDataset(image_path='/public/share_dataset/chc/robonet_frames',transform=row_image_transform,cut=None,mode='eval')	



	sampler=DistributedSampler(train_dataset)
	eval_sampler=DistributedSampler(eval_dataset)
	# while True:
		# if subdir == 1 and block_id < len(lr_schedule):
		# 	model.set_optimizer(config.lr * lr_schedule[block_id])
		
		# train_dataset = get_train_dataset(subdir, block_id)
		# train_dataset = ssv2Dataset(image_path='/home/chc/dataset/ssv2_extracted_frames_5',transform=row_image_transform,cut=None)
	train_epoch(model, train_dataset,optimizer,sampler,eval_dataset,eval_sampler)
		# vqvae_train_epoch(model, train_dataset,optimizer,sampler)
		# del train_dataset
		
		# model.module.save()
		
		# tune_dataset = get_tune_dataset()
		# # state_reconstruct_test(model, tune_dataset)
		# action_regress_test(model, tune_dataset)
		# del tune_dataset
		
		# block_id += 1
		# if block_id == 50:
		# 	subdir += 1
		# 	block_id = 0
		# if subdir == 5:
		# 	subdir = 1


if __name__ == '__main__':
	start=time.time()
	pretrain()
	end=time.time()
	print('run time:',end-start)

# torchrun --nproc_per_node=8 pretrain.py