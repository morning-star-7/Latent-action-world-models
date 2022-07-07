import lpips
import torch
import matplotlib.pyplot as plt
from test import show_image_metric
import omegaconf
import hydra
import torchvision.transforms as T
import numpy as np
from PIL import Image
from r3m import load_r3m
from sklearn.metrics.pairwise import cosine_similarity
import os


# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_squeeze = lpips.LPIPS(net='squeeze')
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
# r3m
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

transforms = T.Compose([
    # T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
    ]) # ToTensor() divides by 255

N=0
S=0
sum_lpips=0
folder_name='./eval_visualization_RR_24_s_1024_42_bs256_adamW_cos_1e-3_standard'
subfolder=os.path.exists(folder_name+'/metric_result_all')
if not subfolder:
	os.makedirs(folder_name+'/metric_result_all')
for i in range(1,4):
    for j in range(8):
        for k in range(32):
            obs0_img=plt.imread(folder_name+'/metric_data_all/obs0_'+str(k)+'_'+str(i)+'cuda:'+str(j)+'.png',format='png')[:,:,:3]
            recon0_img=plt.imread(folder_name+'/metric_data_all/recon0_'+str(k)+'_'+str(i)+'cuda:'+str(j)+'.png',format='png')[:,:,:3]
            obs1_img=plt.imread(folder_name+'/metric_data_all/obs1_blur_'+str(k)+'_'+str(i)+'cuda:'+str(j)+'.png',format='png')[:,:,:3]
            recon1_img=plt.imread(folder_name+'/metric_data_all/recon1_'+str(k)+'_'+str(i)+'cuda:'+str(j)+'.png',format='png')[:,:,:3]
            obs0_img_t=torch.from_numpy(obs0_img)
            recon0_img_t=torch.from_numpy(recon0_img)
            obs1_img_t=torch.from_numpy(obs1_img)
            recon1_img_t=torch.from_numpy(recon1_img)
            obs0_img_int=(obs0_img*255).astype(np.uint8)
            recon0_img_int=(recon0_img*255).astype(np.uint8)
            obs1_img_int=(obs1_img*255).astype(np.uint8)
            recon1_img_int=(recon1_img*255).astype(np.uint8)
            obs0=torch.from_numpy(obs0_img).unsqueeze(0)
            recon0=torch.from_numpy(recon0_img).unsqueeze(0)
            obs1=torch.from_numpy(obs1_img).unsqueeze(0)
            recon1=torch.from_numpy(recon1_img).unsqueeze(0)
            obs0 = torch.einsum('nhwc->nchw', obs0)
            recon0 = torch.einsum('nhwc->nchw', recon0)
            obs1 = torch.einsum('nhwc->nchw', obs1)
            recon1 = torch.einsum('nhwc->nchw', recon1)
            # plt.imsave('./test_save.png',obs0)
            d_0 = loss_fn_vgg(obs1, recon0).item()
            d_1 = loss_fn_vgg(obs1, recon1).item()
            d_0=round(d_0,3)
            d_1=round(d_1,3)
            sum_lpips+=d_1
            if d_1<d_0:
                S+=1
            # else:
            #     print('fail case:',i)
            N+=1


            # ## ENCODE IMAGE
            # # image = np.random.randint(0, 255, (500, 500, 3))
            # # print(image.dtype)
            # # print(obs0_img_int)
            # # image=obs0_img_int
            # # plt.imsave('./int_img.png',obs0_img_int)
            # preprocessed_image_obs0 = transforms(Image.fromarray(obs0_img_int.astype(np.uint8))).reshape(-1, 3, 224, 224).to(device)
            # preprocessed_image_recon0 = transforms(Image.fromarray(recon0_img_int.astype(np.uint8))).reshape(-1, 3, 224, 224).to(device)
            # preprocessed_image_obs1 = transforms(Image.fromarray(obs1_img_int.astype(np.uint8))).reshape(-1, 3, 224, 224).to(device)
            # preprocessed_image_recon1 = transforms(Image.fromarray(recon1_img_int.astype(np.uint8))).reshape(-1, 3, 224, 224).to(device)
            # # preprocessed_image.to(device) 
            # # print(preprocessed_image.mean())
            # with torch.no_grad():
            #     embedding_obs0 = r3m(preprocessed_image_obs0 * 255.0) ## R3M expects image input to be [0-255]
            #     embedding_recon0 = r3m(preprocessed_image_recon0 * 255.0)
            #     embedding_obs1 = r3m(preprocessed_image_obs1 * 255.0)
            #     embedding_recon1 = r3m(preprocessed_image_recon1 * 255.0)
            # # print(embedding.shape) # [1, 2048]
            # # d_0_cos=cosine_similarity(embedding_obs1,embedding_recon0).item()
            # # d_1_cos=cosine_similarity(embedding_obs1,embedding_recon1).item()
            # d_0_cos=((embedding_obs1-embedding_recon0)**2).sum().item()
            # d_1_cos=((embedding_obs1-embedding_recon1)**2).sum().item()
            # d_0_cos=round(d_0_cos,3)
            # d_1_cos=round(d_1_cos,3)


            plt.subplot(1, 4, 1)
            show_image_metric(obs0_img_t, "obs_0")
            plt.subplot(1, 4, 2)
            show_image_metric(recon0_img_t, "PL_24:"+str(d_0))
            plt.subplot(1, 4, 3)
            show_image_metric(obs1_img_t, "ref")
            plt.subplot(1, 4, 4)
            show_image_metric(recon1_img_t, "PL24:"+str(d_1))
            plt.show()
            plt.savefig(folder_name+'/metric_result_all/'+str(k)+'_'+str(i)+str(j)+'.png')
            # plt.savefig('./eval_visualization_48_42/test_4_42_'+str(cnt)+'.png')
            plt.close()

print('success:',S)
print('total:',N)
print('mean lpips:',sum_lpips/N)



