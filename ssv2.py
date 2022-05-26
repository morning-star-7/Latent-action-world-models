import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import numpy as np
from torchvision.io import read_image
from config import config
from torchvision import transforms
from matplotlib import pyplot as plt
import shutil

class ssv2Dataset(Dataset):
    def __init__(self,image_path,transform=None, cut=None):
        self.image_path=image_path
        self.transform=transform
        self.cut=cut
        self.files=os.listdir(self.image_path)
        if self.cut is not None:
            self.files=self.files[:self.cut]

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sub_file=self.files[index]
        image_file=os.path.join(self.image_path,sub_file)
        images_path=os.listdir(image_file)
        obs0 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        obs1 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        for i in range(1):
            obs0_img=read_image(os.path.join(image_file,images_path[i]))
            obs1_img=read_image(os.path.join(image_file,images_path[i+4]))
            obs01=torch.cat([obs0_img,obs1_img],dim=0)
            # print(obs01.shape)
            obs01=self.transform(obs01).numpy()
            # obs0_img=self.transform(obs0_img).numpy()
            # obs1_img=self.transform(obs1_img).numpy()
            obs0_img=obs01[0:3]
            obs1_img=obs01[3:6]
            obs0[i*3:i*3+3]=obs0_img
            obs1[i*3:i*3+3]=obs1_img

        return obs0, obs1


# def main():
#     row_image_transform = transforms.Compose([
# 		transforms.RandomCrop(224,pad_if_needed=True)
# 	])
#     image_path='/home/chc/dataset/ssv2_extracted_frames_5'
#     dataset=ssv2Dataset(image_path=image_path,transform=row_image_transform,cut=None)
#     for i in range(len(dataset)):
#         obs0, obs1=dataset[i]
#         print(i)
#     print(obs1.shape)
#     print(len(dataset))
#     quit()
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#     axs[0, 0].imshow(obs0[0:3].T)
#     axs[0, 1].imshow(obs1[0:3].T)
#     axs[1, 0].imshow(obs0[0:3].T)
#     axs[1, 1].imshow(obs1[0:3].T)
#     fig.savefig('visualize_ss.png')


# if __name__=='__main__':
#     main()

# ataset:',len(dataset))
#     # file_path=os.listdir(image_path)
#     # # print(file_path[100000])
#     # for i in range(len(file_path)):
#     #     images_list=os.path.join(image_path,file_path[i])
#     #     images=os.listdir(images_list)
#     #     if len(images)<10:
#     #         print(images_list)
#     #         shutil.rm