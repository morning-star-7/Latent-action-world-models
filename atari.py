import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
import gzip
import torch
from torch import nn
from typing import List
from config import config
from torch.utils.data import Dataset, DataLoader
from tools import bisect


class AtariDataset(Dataset):
	def __init__(self, subdir=1, block_id=1, cut=None):
		self.path = '/home/chc/dataset/Breakout/'
		self.subdir = subdir
		self.block_id = block_id
		
		self.observation_buffer: np.ndarray
		self.action_buffer: np.ndarray
		self.reward_buffer: np.ndarray
		self.terminal_buffer: np.ndarray
		self.terminal_idx: List
		
		# self.mode = 'origin'
		self.cut = cut
		self._load_buffer()
	
	def _load_buffer(self):
		print('GameLoader:_load_buffer (%d, %d)' % (self.subdir, self.block_id))
		path = self.path + '%d/replay_logs/' % self.subdir
		# with gzip.open(path + '$store$_observation_small_ckpt.%d.gz' % self.block_id, 'rb') as f:
		# if self.mode == 'small':
		# 	with gzip.open(path + '$store$_observation_small_ckpt.%d.gz' % self.block_id, 'rb') as f:
		# 		self.observation_buffer = np.frombuffer(f.read(), dtype=np.uint8)
		# else:
		with gzip.open(path + '$store$_observation_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.observation_buffer = np.load(f)
		print(self.observation_buffer.shape)
		self.observation_buffer = self.observation_buffer.reshape((1000000, *config.observation_shape[1:]))
		with gzip.open(path + '$store$_action_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.action_buffer = np.load(f)
		with gzip.open(path + '$store$_reward_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.reward_buffer = np.load(f)
		with gzip.open(path + '$store$_terminal_ckpt.%d.gz' % self.block_id, 'rb') as f:
			self.terminal_buffer = np.load(f)
		
		if self.cut is not None:
			self.observation_buffer = self.observation_buffer[: self.cut]
			self.action_buffer = self.action_buffer[: self.cut]
			self.reward_buffer = self.reward_buffer[: self.cut]
			self.terminal_buffer = self.terminal_buffer[: self.cut]
		
		self.terminal_idx = [-1] + list(np.where(self.terminal_buffer == 1)[0])
		# print(self.terminal_idx)
		
		self.buffer_length = self.observation_buffer.shape[0]
		print('GameLoader:_load_buffer done. with buffer_length = %d' % self.buffer_length)
	
	def __len__(self):
		return self.buffer_length * config.max_cd
	
	def get_stack_num(self, i, frame_stack):
		pre_terminal_idx = self.terminal_idx[bisect(self.terminal_idx, i - 1)]
		# print('jb', i, pre_terminal_idx)
		return min(frame_stack, i - pre_terminal_idx)
	
	def __getitem__(self, i_cd):
		# close, sharp
		# i = self.idx_dict[item]
		
		i, cd = i_cd // config.max_cd, i_cd % config.max_cd
		if not config.zero_cd:
			cd += 1
		
		if i + cd >= self.buffer_length or self.terminal_buffer[i: i + cd].any():
			# illegal, randomly return a new item.
			return self[np.random.randint(0, len(self))]
		
		obs0 = np.zeros((config.frame_stack, *config.observation_shape[1:]), dtype=np.uint8)
		obs1 = np.zeros((config.frame_stack, *config.observation_shape[1:]), dtype=np.uint8)
		
		stack_num0 = self.get_stack_num(i, config.frame_stack)
		stack_num1 = self.get_stack_num(i + cd, config.frame_stack)
		
		# print(i, i + cd, stack_num0, stack_num1)
		
		obs0[-stack_num0:] = self.observation_buffer[i - stack_num0 + 1: i + 1]
		obs1[-stack_num1:] = self.observation_buffer[i + cd - stack_num1 + 1: i + cd + 1]
		
		action = np.zeros(config.max_cd, dtype=np.int32)
		reward = np.zeros(config.max_cd, dtype=np.int32)
		r = cd + 1 if config.zero_cd else cd
		action[:r] = self.action_buffer[i: i + r]
		reward[:r] = self.reward_buffer[i: i + r]
		
		return (obs0, obs1), action, reward


class AtariDatasetMultistep(AtariDataset):
	def __init__(self, subdir=1, block_id=1, cut=None):
		super(AtariDataset, self).__init__(subdir, block_id, cut)
	
	def __len__(self):
		return self.buffer_length
	
	def __getitem__(self, i):
		# close, sharp
		# i = self.idx_dict[item]
		T = config.max_dynamic_timestep
		
		if self.terminal_buffer[i: i + T + 1].any():  # TODO ??
			# illegal, randomly return a new item.
			return self[np.random.randint(0, len(self))]
		
		obs = np.zeros((T + 1, config.frame_stack, *config.observation_shape[1:]), dtype=np.uint8)
		
		for t in range(T + 1):
			stack_num = self.get_stack_num(i + t, config.frame_stack)
			obs[-stack_num:] = self.observation_buffer[i + t - stack_num + 1: i + t + 1]
		
		action = np.zeros(T, dtype=np.int32)
		reward = np.zeros(T, dtype=np.int32)
		action = self.action_buffer[i: i + T]
		reward = self.reward_buffer[i: i + T]
		
		return obs, action, reward


def main():
	dataset = AtariDataset(1, 0)
	print(dataset.action_buffer)
	
	for i in range(len(dataset)):
		(obs0, obs1), action, reward = dataset[i]
		# print(action)
	print(obs0.shape)
	print(obs0.type)
	print(obs0.size())


# print(obs0.shape, obs1.shape, action.shape, reward.shape)


if __name__ == '__main__':
	main()
