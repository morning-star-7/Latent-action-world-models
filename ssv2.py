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
import cv2
import skvideo.io

class ssv2Dataset(Dataset):
    def __init__(self,image_path,transform=None, mode='train',cut=None):
        self.image_path=image_path
        self.transform=transform
        self.cut=cut
        self.mode=mode
        self.files=os.listdir(self.image_path)
        if self.cut is not None:
            self.files=self.files[:self.cut]
        if self.mode=='train':
            self.files=self.files[:200000]
        elif self.mode=='eval':
            self.files=self.files[-1000:]

    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sub_file=self.files[index]
        image_file=os.path.join(self.image_path,sub_file)
        images_path=os.listdir(image_file)
        obs0 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        obs1 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        for i in range(1):
            # obs0_img=read_image(os.path.join(image_file,images_path[i]))
            # obs1_img=read_image(os.path.join(image_file,images_path[i+4]))
            obs0_img=torch.from_numpy(plt.imread(os.path.join(image_file,images_path[i])))
            obs1_img=torch.from_numpy(plt.imread(os.path.join(image_file,images_path[i+4])))
            obs0_img=torch.einsum('hwc->chw', obs0_img)
            obs1_img=torch.einsum('hwc->chw', obs1_img)
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





class ego4dDataset(Dataset):
    def __init__(self,image_path,transform=None, mode='train',cut=None):
        self.image_path=image_path
        self.transform=transform
        self.cut=cut
        self.mode=mode
        self.subfiles=os.listdir(self.image_path)
        self.files=[]
        for i,subfile in enumerate(self.subfiles):
            full_file=os.path.join(self.image_path,subfile)
            images=os.listdir(full_file)
            for j,image in enumerate(images):
                full_images=os.path.join(full_file,image)
                self.files.append(full_images)




        if self.cut is not None:
            self.files=self.files[:self.cut]
        if self.mode=='train':
            self.files=self.files[:130000]
        elif self.mode=='eval':
            self.files=self.files[-1000:]



    
    def __len__(self):
        return len(self.files)-1

    def __getitem__(self, index):
        # sub_file=self.files[index]
        # image_file=os.path.join(self.image_path,sub_file)
        # images_path=os.listdir(image_file)
        img_0=self.files[index]
        img_1=self.files[index+1]
        obs0 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        obs1 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        for i in range(1):
            # obs0_img=read_image(os.path.join(image_file,images_path[i]))
            # obs1_img=read_image(os.path.join(image_file,images_path[i+4]))
            obs0_img=torch.from_numpy(plt.imread(img_0))
            obs1_img=torch.from_numpy(plt.imread(img_1))
            obs0_img=torch.einsum('hwc->chw', obs0_img)
            obs1_img=torch.einsum('hwc->chw', obs1_img)
            if obs0_img.shape != obs1_img.shape:
                obs1_img=obs0_img

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













class ssv2VideoDataset(Dataset):
    def __init__(self,image_path,transform=None, mode='train',cut=None):
        self.image_path=image_path
        self.transform=transform
        self.cut=cut
        self.mode=mode
        self.files=os.listdir(self.image_path)
        if self.cut is not None:
            self.files=self.files[:self.cut]
        if self.mode=='train':
            self.files=self.files[:200000]
        elif self.mode=='eval':
            self.files=self.files[-1000:]
            # self.files=self.files[:200000]


    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        video_idx=self.files[index]
        video=cv2.VideoCapture(os.path.join(self.image_path,video_idx))
        num_frames=video.get(cv2.CAP_PROP_FRAME_COUNT)
        # sub_file=self.files[index]
        # image_file=os.path.join(self.image_path,sub_file)
        # images_path=os.listdir(image_file)
        obs0 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        obs1 = np.zeros((config.ss_frame_stack, *config.ss_observation_shape[1:]), dtype=np.uint8)
        for i in range(1):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_0,frame_0=video.read()
            if success_0==False:
                print('load data error frame 0')
                print(os.path.join(self.image_path,video_idx))
                quit()
            if num_frames>12:
                video.set(cv2.CAP_PROP_POS_FRAMES,5)
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES,5)
            success_1,frame_1=video.read()
            # print(type(frame_1))
            if success_1==False:
                frame_1=frame_0

            # videoarray=skvideo.io.vreader(os.path.join(self.image_path,video_idx))
            # for j,frame in enumerate(videoarray):
            #     if j==0:
            #         frame_0=frame
            #     if j==5:
            #         frame_1=frame
            #         break
            # frame_0=videoarray[0]
            # frame_1=videoarray[1]
            obs0_img=torch.from_numpy(frame_0)
            obs1_img=torch.from_numpy(frame_1)
            # print(type(obs1_img))
            obs0_img=torch.einsum('hwc->chw', obs0_img)
            obs1_img=torch.einsum('hwc->chw', obs1_img)
            obs01=torch.cat([obs0_img,obs1_img],dim=0)
            # print(obs01.shape)
            obs01=self.transform(obs01).numpy()
            # obs0_img=self.transform(obs0_img).numpy()
            # obs1_img=self.transform(obs1_img).numpy()
            obs0_img=obs01[0:3]
            obs1_img=obs01[3:6]
            obs0[i*3:i*3+3]=obs0_img
            obs1[i*3:i*3+3]=obs1_img
            # print(type(obs0))
            # quit()

        return obs0, obs1




def main():
    row_image_transform = transforms.Compose([
		transforms.RandomCrop(224,pad_if_needed=True)
	])
    # image_path='/public/share_dataset/ssv2_extracted_frames_5'
    # dataset=ssv2Dataset(image_path=image_path,transform=row_image_transform,cut=None)
    # image_path="/public/MARS/datasets/ssv2/20bn-something-something-v2"
    # dataset=ssv2Dataset(image_path=image_path,transform=row_image_transform,mode='train',cut=None)
    image_path="/public/share_dataset/toy_ego4d/frames"
    dataset=ego4dDataset(image_path=image_path,transform=row_image_transform,mode='eval',cut=None)
    # for i in range(len(dataset)):
    #     obs0, obs1=dataset[i]
    #     print(i)
    # print(obs1.shape)
    obs0, obs1=dataset[10]
    print(obs0.shape)
    # quit()
    print(len(dataset))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(obs0[0:3].T)
    axs[0, 1].imshow(obs1[0:3].T)
    axs[1, 0].imshow(obs0[0:3].T)
    axs[1, 1].imshow(obs1[0:3].T)
    fig.savefig('visualize_ss.png')


if __name__=='__main__':
    main()

# ataset:',len(dataset))
#     # file_path=os.listdir(image_path)
#     # # print(file_path[100000])
#     # for i in range(len(file_path)):
#     #     images_list=os.path.join(image_path,file_path[i])
#     #     images=os.listdir(images_list)
#     #     if len(images)<10:
#     #         print(images_list)
#     #         shutil.rm