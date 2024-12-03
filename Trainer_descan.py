import warnings

# Ignore all warnings of any kind
warnings.filterwarnings("ignore")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import reduction
from tqdm import tqdm  # Make sure to import tqdm correctly
import numpy as np
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
import natsort
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter,ImageOps
import os
import os.path
from pathlib import Path
from os import listdir
from os.path import join
from matplotlib import cm
import sys
from scipy.io import loadmat
import random
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct,idct
from skimage import filters
from utils_image import *


class PSNR:

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

psnr = PSNR()

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader ,random_split
from torch.nn import functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim
import tqdm
from skimage.transform import rotate

MODEL_ = 'DD256_R34_CBAM_2_20241203'
log_name = './logs_val/logger_'+MODEL_+'.txt'
print('Starting the logger',  file=open(log_name, 'w'))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def imshow(inp, title=None):
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    plt.axis('off')
    #inp=np.int 
    #print(inp.shape)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=None)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class Custom_Loader(Dataset):
    def __init__(self,parent_dir,min_beta,max_beta,steps, distort_threshold,schedule='LINEAR'):
        



        self.clean_path = join(parent_dir,'clean_512')
        self.dir_clean,self.dir_scan= self.image_paths(parent_dir)
        self.min_beta= min_beta
        self.max_beta = max_beta
        self.steps = steps
        self.schedule = schedule
        self.distort_threshold = distort_threshold
        self.beta = self.beta_schedule(self.min_beta,self.max_beta,self.steps,self.schedule)
        self.alpha = self.alpha_t(self.beta)
        self.alpha_hat = self.commulative_alpha(self.alpha)
        # self.t_steps = np.logspace(0,sigma,self.steps,base=10)
        self.Transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(512) # 512
                                              ])

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"])

 

    def random_file(self,folder_path):
        files = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                files.append(file_path)
        return random.choice(files)

    def degrade_chroma(self, image, factor):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = s.astype(np.float32)
        s *= factor
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def add_gaussian_noise(self, image, mean, sigma):
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
        return noisy_image

    def random_perspective_transform(self, image, max_warp=50, interp=cv2.INTER_CUBIC):
        height, width, _ = image.shape
        warp = max_warp * np.random.uniform(size=(8,)) - max_warp / 2
        # The four source points
        s_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        # The four destination points
        d_points = np.float32([[warp[0], warp[1]],
                            [width + warp[2], warp[3]],
                            [warp[4], height + warp[5]],
                            [width + warp[6], height + warp[7]]])
        # Compute the perspective transform
        M = cv2.getPerspectiveTransform(s_points, d_points)
        # Apply the perspective transform to the image
        transformed = cv2.warpPerspective(image, M, (width, height), flags=interp, borderMode=cv2.BORDER_REPLICATE)
        return transformed


    def load_img(self,img_path):
        img = Image.open(img_path)
        # img = img.convert('RGB')
        # img = np.array(img).astype(np.float32)/255
        # plt.imshow(img)
        return img #torch.tensor(img)
    
    def forward_process_step(self,x_o,alpha_hat,time_step):
        # x_shape = x_o.shape
        noise = torch.randn_like(x_o)
        xt = (torch.sqrt(alpha_hat[time_step-1])*x_o) + (torch.sqrt(1-alpha_hat[time_step-1])*noise)
        return xt,noise


    def beta_schedule(self,min_beta,max_beta,steps,schedule='LINEAR'):
        if schedule =='LINEAR':
            beta = torch.linspace(min_beta,max_beta,steps)
        return beta

    def normalize_to_neg_one_to_one(self,img):
        return img * 2 - 1

    def unnormalize_to_zero_to_one(self,t):
        return (t + 1) * 0.5

    def alpha_t(self,beta):
        alpha = 1-beta
        return alpha

    def commulative_alpha(self,alpha):
        alpha_hat = torch.cumprod(alpha,dim=0)
        return alpha_hat


    
    def image_paths(self,parent_dir):

        dir_clean = []
        dir_scan = []
        iii = 0
        for x in  natsort.natsorted(listdir(join(parent_dir,'clean_512')),reverse=True):
            # if iii >= 0 and iii < 1000: 
            if os.path.isfile(join(parent_dir,'clean_512',x)) and os.path.isfile(join(parent_dir,'scan_512',x)):#color, train_B
                dir_clean.append(join(parent_dir,'clean_512', x))
                dir_scan.append(join(parent_dir, 'scan_512', x))
            # iii += 1 
        return dir_clean,dir_scan
    
    def __len__(self):
        return len(self.dir_scan)




    def mean_std_image(self, image):
        # Assuming image is a PyTorch tensor of shape [C, H, W] and dtype torch.float32
        # Calculate the mean and std dev for each channel
        mean = torch.mean(image, dim=[1, 2])
        std = torch.std(image, dim=[1, 2])
        
        return mean, std

    def color_shift_function(self, scan_image, mean_clean, std_clean, mean_scan, std_scan):
        # Assuming scan_image is a PyTorch tensor of shape [C, H, W] and dtype torch.float32
        # Normalize scan image
        normalized_scan = (scan_image - mean_scan[:, None, None]) / std_scan[:, None, None]
        
        # Shift colors based on clean image stats
        shifted_scan = normalized_scan * std_clean[:, None, None] + mean_clean[:, None, None]
        
        # Normalize the shifted image to have values between 0 and 1
        shifted_scan = shifted_scan - shifted_scan.min()
        shifted_scan = shifted_scan / shifted_scan.max()
        
        return shifted_scan




    def __getitem__(self, index):
        np.random.seed()
        img_clean = self.load_img(self.dir_clean[index])
        img_scan = self.load_img(self.dir_scan[index])
        blending_target = self.random_file(self.clean_path)
        img_blend = self.load_img(blending_target)


        hflip = random.randint(0,1)
        noise = random.randint(0,1)
#         print(hflip)

        persp_trans = random.random() > 0.5
        mirr  = random.random() > 0.5

        if hflip:
            img_clean = ImageOps.flip(img_clean)
            img_scan = ImageOps.flip(img_scan)

        if mirr:
            img_clean = ImageOps.mirror(img_clean)
            img_scan = ImageOps.mirror(img_scan)

        

        img_clean = np.array(img_clean.convert('RGB')).astype(np.uint8)
        img_scan = np.array(img_scan.convert('RGB')).astype(np.uint8)
        img_blend = np.array(img_blend.convert('RGB')).astype(np.uint8)

        distort_prob = random.uniform(0, 1)

        distort_threshold = self.distort_threshold



        if distort_prob <distort_threshold:                         ### MIX VS CLEAN PERCENTAGE
            alpha = random.uniform(0.85, 0.95)

            threshold_blended = random.uniform(0, 1)

            if threshold_blended <= 0.7:
                blended = cv2.addWeighted(img_clean, alpha, img_blend, (1-alpha), 0) 
            else:
                blended = img_clean
            ##### Misalignment distortion #####

            threshold_perspective = random.uniform(0, 1)

            if threshold_perspective <= -2:
                random_perspective = self.random_perspective_transform(blended, max_warp = random.uniform(20,50))
            else:
                random_perspective = blended

            ##### Color transition #####

            threshold_chroma = random.uniform(0, 1)

            if threshold_chroma <= 0.8:
                degraded_image = self.degrade_chroma(random_perspective, random.uniform(0.2, 0.4))
            else:
                degraded_image = self.degrade_chroma(random_perspective, random.uniform(1.2, 1.5)) 

            ##### Noise and Halftone pattern by Gaussian noise #####

            noise_img = self.add_gaussian_noise(degraded_image, mean=np.random.randint(5,10), sigma=np.random.randint(10,20))

            ##### Linear laser pattern ##### 

            h, w = noise_img.shape[:2]

            threshold_laser = random.uniform(0, 1)

            if threshold_laser <= 0.5:
                # Draw the pattern on the image along the horizontal direction
                num_horizontal = random.randint(1, 5)
                for i in range(num_horizontal):
                    y = random.randint(0, h)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.line(noise_img, (0, y), (w, y), color, 1)

                # Draw the pattern on the image along the vertical direction
                num_vertical = random.randint(1, 5)
                for i in range(num_vertical):
                    x = random.randint(0, w)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    cv2.line(noise_img, (x, 0), (x, h), color, 1)
            else:
                noise_img = noise_img
 
            ##### Scratch distortion ##### 

            threshold_scratch = random.uniform(0, 1)

            if threshold_scratch <= -0.1: # Not scratch
                mask_scratch = np.zeros(noise_img.shape[:2], dtype=np.uint8)
                for i in range(200):
                    start = (random.randint(0, noise_img.shape[0]), random.randint(0, noise_img.shape[1]/2))
                    end = (random.randint(0, noise_img.shape[0]), random.randint(0, noise_img.shape[1]/2))
                    thickness = random.randint(1, 3)
                    color = random.randint(200,255)
                    cv2.line(mask_scratch, start, end, color, thickness)

                mask_scratch = cv2.bitwise_not(mask_scratch)
                output_img = cv2.bitwise_and(noise_img, noise_img, mask=mask_scratch)
            else:
                output_img = noise_img

            ##### Dust distortion ##### 

            threshold_dust = random.uniform(0, 1)

            if threshold_dust <= 0.3:
                mask_dust = np.zeros(noise_img.shape[:2], dtype=np.uint8)
                for i in range(200):
                    center = (random.randint(0, noise_img.shape[0]), random.randint(0, noise_img.shape[1]))
                    radius = random.randint(1, 5)
                    color = random.randint(200, 255)
                    cv2.circle(mask_dust, center, radius, color, thickness=-1)

                mask_dust = cv2.bitwise_not(mask_dust)
                output_img = cv2.bitwise_and(output_img, output_img, mask=mask_dust)
            else:
                output_img = output_img
            img_scan = output_img

        img_clean = uint2single(img_clean)
        img_scan = uint2single(img_scan)


        img_clean = self.Transforms(np.array(img_clean))
        img_scan = self.Transforms(np.array(img_scan))

        mean_clean, std_clean = self.mean_std_image(img_clean)
        mean_scan, std_scan = self.mean_std_image(img_scan)

        img_scan_shift = color_shift_function(img_scan,mean_clean,std_clean, mean_scan, std_scan)
        
       
        r = img_scan_shift[0,:,:]
        g = img_scan_shift[1,:,:]
        b = img_scan_shift[2,:,:]
        
        # torchvision.utils.save_image(img_scan_shift,"./check.png")

        img_clean = normalize_to_neg_one_to_one(img_clean)
        img_scan_shift = normalize_to_neg_one_to_one(img_scan_shift)

        r_mean, g_mean, b_mean = torch.mean(r), torch.mean(g), torch.mean(b)
        r_std, g_std, b_std = torch.std(r), torch.std(g), torch.std(b)
        
        color_dist =  torch.tensor([r_mean, g_mean, b_mean,r_std, g_std, b_std])        
        
        

        t = random.randint(1,self.steps)

        x_t,noise = self.forward_process_step(img_clean,self.alpha_hat,t)
        x_t_scan = torch.cat((x_t,img_scan_shift),0)

        return x_t_scan,noise,t,color_dist




def mean_std_image(image):
    # Assuming image is a PyTorch tensor of shape [C, H, W] and dtype torch.float32
    # Calculate the mean and std dev for each channel
    mean = torch.mean(image, dim=[1, 2])
    std = torch.std(image, dim=[1, 2])
    
    return mean, std

def color_shift_function(scan_image, mean_clean, std_clean, mean_scan, std_scan):
    # Assuming scan_image is a PyTorch tensor of shape [C, H, W] and dtype torch.float32
    # Normalize scan image
    normalized_scan = (scan_image - mean_scan[:, None, None]) / std_scan[:, None, None]
    
    # Shift colors based on clean image stats
    shifted_scan = normalized_scan * std_clean[:, None, None] + mean_clean[:, None, None]
    
    # Normalize the shifted image to have values between 0 and 1
    shifted_scan = shifted_scan - shifted_scan.min()
    shifted_scan = shifted_scan / shifted_scan.max()
    
    return shifted_scan



Transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(512)
                                        ])
from tqdm import tqdm


def sampling(denoising_function, img_scan, color_dist, initial_state, alpha, alpha_hat, steps):
    img_scan = img_scan.to(device)
    color_dist = color_dist.to(device)
    x_t = initial_state

    for step in tqdm(reversed(range(steps)), desc='Sampling Process', total=steps):
        a_t = alpha[step].to(device)
        a_t_hat = alpha_hat[step].to(device)
        if len(x_t.shape) > 3:
            inp = torch.cat((x_t[0], img_scan), 0).to(device)
        else:
            inp = torch.cat((x_t, img_scan), 0).to(device)
        if step > 0:
            x_t = (x_t - (((1 - a_t) / torch.sqrt(1 - a_t_hat)) * denoising_function(inp[None, ...], torch.tensor([step + 1], dtype=torch.float32, device=device), color_dist[None, ...]))) + torch.sqrt(1 - a_t) * torch.randn_like(x_t).to(device)
        else:
            x_t = (x_t - (((1 - a_t) / torch.sqrt(1 - a_t_hat)) * denoising_function(inp[None, ...], torch.tensor([step + 1], dtype=torch.float32, device=device), color_dist[None, ...])))

    return x_t


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def validate(test_folder, model_function, color_encoder, custom_set, steps, epoch):
    
    newpath = './Validation/DD256_R34_CBAM_2_20241104/' + str(epoch)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    total_PSNR = []
    print('Validation ' + str(epoch), file=open(log_name, 'a'))
    for x in natsort.natsorted(listdir(join(test_folder, 'scan_512')), reverse=True):
        
        
        img_scan = Transforms(Image.open(join(test_folder, 'scan_512', x)))
        img_clean = Transforms(Image.open(join(test_folder, 'clean_512', x)))
        

  
        output_dist = color_encoder(img_scan[None,...].to(device))
        [r_mean, g_mean, b_mean,r_std, g_std, b_std]= output_dist[0].cpu()
        
        mean_pred = torch.tensor([r_mean, g_mean, b_mean])
        std_pred = torch.tensor([r_std, g_std, b_std])



        mean_scan,std_scan = mean_std_image(img_scan)
        # print(mean_scan.shape, mean_pred.shape)

        image_scan_shift = color_shift_function(img_scan,mean_pred,std_pred,mean_scan,std_scan)    
        
        color_dist = torch.tensor([r_mean, g_mean, b_mean, r_std, g_std, b_std])

        image_scan_shift = normalize_to_neg_one_to_one(image_scan_shift)
        
        # Preparing alpha and alpha_hat from custom_set
        alpha = (custom_set.alpha).to(device)
        alpha_hat = (custom_set.alpha_hat).to(device)
        initial_state = (image_scan_shift).to(device)
        # initial_state = torch.randn_like(image_scan_shift).to(device)

        # Sampling process
        Descanned = sampling(model_function, image_scan_shift, color_dist, initial_state, alpha, alpha_hat, steps).clamp_(-1, 1)
        Descanned = unnormalize_to_zero_to_one(Descanned)
        
        # PSNR Calculation
        current_psnr = psnr(Descanned[0], img_clean.to(device))
        print(f'Name: {x}, PSNR: {current_psnr.item()}', file=open(log_name, 'a'))
        total_PSNR.append(current_psnr.item())
        
        # Save the Descanned image
        # torchvision.utils.save_image(Descanned, os.path.join(newpath, f'{x}_{epoch + 1}.png'))

    avg_psnr = np.mean(total_PSNR)
    print(f'Average Validation PSNR: {avg_psnr}')
    print(f'Average Validation PSNR: {avg_psnr}', file=open(log_name, 'a'))



def train_model(model, color_encoder, optimizer,criterion,epoch_done,test_folder, train_set, custom_set, log_name,num_epochs=3):
    loss_train=[]
    Epochs=[]
    total_iteration = 0
    for epoch in range(epoch_done+1,num_epochs):
        print('Epoch ' ,epoch+1,'/',num_epochs)
        print('-' * 10)
        model.train()
        running_loss = 0.0
        i=0
        for x_t_scan,noise,t,color_dist in train_set:
                total_iteration = total_iteration
                x_t_scan = x_t_scan.to(device)#.float()
                # print(color_dist)
                # print(torch.max(x_t_lr[:,3::]))
                color_dist = color_dist.to(device)
                noise = noise.to(device)#.float()
                t = t.to(device)#.float()
                i=i+1
                # print(inputs.shape)
                optimizer.zero_grad()
                # out = model(x_t_scan,t,color_dist)
                out = model(x_t_scan,t,color_dist)
                loss=criterion(out,noise)
                print('\r ',i,'  ','loss  ',loss,end='')
                loss.backward()
                
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.detach() * noise.size(0)

        epoch_loss = running_loss / len(custom_set)

        
                
        Epochs.append(epoch)
        loss_train.append(epoch_loss)

        print('Epoch Loss',epoch_loss.item())   
        print('EPOCH '+ str(epoch)+'  Epoch Loss',str(epoch_loss.item())+' Lr rate '+ str(optimizer.param_groups[0]['lr']),  file=open(log_name, 'a'))



        if  (epoch+1)%10==0:
            model.eval()
            with torch.no_grad():
                validate(test_folder,model, color_encoder, custom_set, 20,epoch)      
            
            # print('    Saved history ')
        if  (epoch+1)%10==0:
            newpath =  './weights_'+MODEL_+'/'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': criterion,
                        }, newpath+'/'+str(epoch+1)+'.pth')
    plt.plot(Epochs,loss_train)   
    return model
