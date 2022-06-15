import numpy as np
import torch
from torch import optim
import argparse
from torch.utils.data import DataLoader

torch.set_printoptions(linewidth=9999999, precision=3, threshold=100000000, sci_mode=False)
np.set_printoptions(linewidth=9999, precision=3)

parser = argparse.ArgumentParser()

# parser.add_argument('-m', '--model',    type=str, default='LSTM')
# parser.add_argument('-n', '--norm',     type=str, default='ID')
parser.add_argument('-s', '--dataset', type=str, default='Breakout')
parser.add_argument('-d', '--device', type=int, default=0)
parser.add_argument('-r', '--restore', type=bool, default=False)
# parser.add_argument('-l', '--lr', type=float, default=0.001)
parser.add_argument('-l', '--lr', type=float, default=5e-4)
# parser.add_argument('-l', '--lr', type=float, default=0.05)
parser.add_argument('-c', '--channel', type=int, default=1000)
# parser.add_argument('-b', '--batch_size', type=int, default=256)
parser.add_argument('-b', '--batch_size', type=int, default=4)
parser.add_argument('-g', '--momentum', type=float, default=0.9)
parser.add_argument('-w', '--weight_decay', type=float, default=0.0001)
# parser.add_argument('-o', '--optimizer', type=str, default='Adam')
parser.add_argument('-o', '--optimizer', type=str, default='SGD')
parser.add_argument('--state_norm', type=bool, default=True)
# parser.add_argument('-t', '--maxT',         type=int, default=100)
# parser.add_argument('-a', '--layers_num',   type=int, default=1)

args = parser.parse_args()


class Config:
	def __init__(self):
		self.batch_size = args.batch_size
		self.channel = args.channel
		self.lr = args.lr
		
		self.momentum = args.momentum
		self.weight_decay = args.weight_decay
		self.clip_max = 1.0
		self.tau = 0.999
		# self.device = 'cuda:0'
		self.device=torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
	
		# self.layers_num = args.layers_num
		
		self.dataset = args.dataset.lower()
		self.observation_shape = (4, 84, 84)
		self.ss_observation_shape = (1*3, 224, 224)
		self.state_shape = (64, 6, 6)
		self.state_size = self.state_shape[1] * self.state_shape[2]
		
		self.latent_action_channel = 64
		self.num_embeddings = 256
		self.latent_dim = 768
		self.state_norm = args.state_norm
		
		self.max_dynamic_timestep = 5
		self.max_cd = 1
		self.zero_cd = False
		self.freeze_encoder_stat = False
		self.bn_momentum = 0.01
		
		self.state_detach = False
		
		self.frame_stack = 4
		self.ss_frame_stack = 1*3
		
		self.representation_loss = 'auto-encoder'
		# self.representation_loss = 'contrastive'
		# self.representation_loss = 'SimSiam'
		
		assert args.optimizer in ['SGD', 'Adam']
		if args.optimizer == 'SGD':
			self.optim = optim.SGD
		if args.optimizer == 'Adam':
			self.optim = optim.Adam
		
		# unused
		self.byol_mt = True
		self.byol_sg = True
		self.byol_pd = True


config = Config()
