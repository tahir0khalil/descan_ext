import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import natsort
from tqdm.auto import tqdm
from Conditional_Reverse_Function import Unet  
from os.path import join
from os import listdir
import numpy as np 
import cv2 
import torch.nn.functional as F 
from FDL_pytorch import FDL_loss

class PSNR:

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

psnr = PSNR()


class ModifiedResNet50(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNet50, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Keep all layers except the final fc
        
        # Define the new layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 6)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to ensure outputs are between 0 and 1
        
    def forward(self, x):
        x = self.base_model(x)  
        x = torch.flatten(x, 1) 
        x = self.fc1(x)  
        x = nn.ReLU()(x)  
        x = self.fc2(x)  
        x = self.sigmoid(x) 
        return x

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fdl_loss = FDL_loss().to(device)

# Model Setup
def load_model_and_encoder(diffusion_weights_path, color_encoder_path):
    # Unet Model
    model = Unet(dim=64, channels=6, out_dim=3, dim_mults=(1, 2, 4, 8, 8)).to(device)
    model = torch.nn.DataParallel(model) #, device_ids=[6])
    model_checkpoint = torch.load(diffusion_weights_path, map_location=device)
    model.load_state_dict(model_checkpoint['model_state_dict'])

    # Color Encoder R34
    color_encoder = models.resnet34(pretrained=False)
    color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False)
    color_encoder = color_encoder.to(device)
    encoder_checkpoint = torch.load(color_encoder_path, map_location=device)
    color_encoder.load_state_dict(encoder_checkpoint)
    
    # Color Encoder R50
    # modified_model = ModifiedResNet50(models.resnet50(pretrained=False, progress=True)) 
    # color_encoder = modified_model.to(device)     
    # encoder_checkpoint = torch.load(color_encoder_path, map_location=device)
    # color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    
    return model, color_encoder

# Image Transformations
Transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512) #512
])

# Beta and Alpha Schedules
def beta_schedule(min_beta=0.0001, max_beta=0.01, steps=2000, schedule='LINEAR'):
    if schedule == 'LINEAR':
        beta = torch.linspace(min_beta, max_beta, steps).to(device)
    # Add other scheduling methods if necessary
    return beta

def prepare_alpha_schedules(beta):
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    return alpha, alpha_hat.to(device)

# Image Normalization Functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# Image Statistics Functions
def mean_std_image(image):
    mean = torch.mean(image, dim=[1, 2])
    std = torch.std(image, dim=[1, 2])
    return mean, std

def color_shift_function(scan_image, mean_clean, std_clean, mean_scan, std_scan):
    normalized_scan = (scan_image - mean_scan[:, None, None]) / std_scan[:, None, None]
    shifted_scan = normalized_scan * std_clean[:, None, None] + mean_clean[:, None, None]
    shifted_scan = shifted_scan - shifted_scan.min()
    shifted_scan = shifted_scan / shifted_scan.max()
    return shifted_scan

# Sampling Function
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

# Test Function
def test(test_folder, model_function, color_encoder, alpha, alpha_hat, steps):
    
    # img save path 
    newpath = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/extension_testing/DD256R34_FDL_Tuner_1k_140' 
    
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    color_encoder.eval()
    model_function.eval()
    SCAN_ = 'scan_512'
    CLEAN_ = 'clean_512' 
    scan_dir = os.path.join(test_folder,SCAN_) # scan_512
    clean_dir = os.path.join(test_folder,CLEAN_) # clean_512

    
    p = [] 
    p_ = [] 
    fdl = [] 
    fdl_down = [] 
    for x in natsort.natsorted(listdir(join(test_folder, SCAN_)), reverse=True): # scan_512
        with torch.no_grad():
            if 'scanner03' in x or 'scanner04' in x: 
                continue
            img_scan = Transforms(Image.open(join(test_folder, SCAN_, x))).to(device) # scan_512
            img_clean = Transforms(Image.open(join(test_folder, CLEAN_, x))).to(device) # clean_512

            output_dist = color_encoder(img_scan[None, ...])
            mean_pred, std_pred = output_dist[0][:3], output_dist[0][3:]
  
            # color encoder
            mean_scan, std_scan = mean_std_image(img_scan)
            image_scan_shift = color_shift_function(img_scan, mean_pred, std_pred, mean_scan, std_scan) 

            
            image_scan_shift = normalize_to_neg_one_to_one(image_scan_shift) # image_scan_shift
            initial_state = image_scan_shift #torch.rand_like(image_scan_shift) 

            Descanned = sampling(model_function, image_scan_shift, output_dist[0], initial_state, alpha, alpha_hat, steps).clamp_(-1, 1)
            Descanned = unnormalize_to_zero_to_one(Descanned) 
  
            downsampled_descanned = F.avg_pool2d(Descanned, kernel_size=2, stride=2) 
  
            downsampled_img_clean = F.avg_pool2d(img_clean.unsqueeze(0), kernel_size=2, stride=2)
            downsampled_img_clean = downsampled_img_clean.squeeze(0) 
            p.append(psnr(img_clean, Descanned))
  
            p_.append(psnr(downsampled_img_clean, downsampled_descanned))
            fdl.append(fdl_loss(img_clean.unsqueeze(0), Descanned))
            fdl_down.append(fdl_loss(downsampled_img_clean.unsqueeze(0), downsampled_descanned))
    print(f"PNSR: {sum(p)/len(p)}")
    print(f"PNSR_: {sum(p_)/len(p_)}")
    print(f"FDL: {sum(fdl)/len(fdl)}")
    print(f"FDL BASE: {sum(fdl_down)/len(fdl_down)}")


# baseline
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights/Descan_diffusion.pth'
# D_CBAM
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_CBAM256_2_20240918/220.pth'
# DD-256 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_D256_FDL_20240929/80.pth'
# DD-256 (CNN)
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_FDL_Tuner_20241016/90.pth'
# DD-256 Tuner R50 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_FDL_Tuner_ONLY_1000_20241021/130.pth'
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_FDL_Tuner_ONLY_1k-2k_20241025/120.pth'
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_FDL_Tuner_ONLY_500_20241026/100.pth'
# DD-256 Tuner R34 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_R34_Tuner_ONLY_1000_20241030/150.pth'
diffusion_weights_path = '/NAS2/tahir/descan_extension/AAAI_Github_Code_Descan/weights_DD256_R34_Tuner_ONLY_1000_20241030/150.pth'


# baseline
color_encoder_path = '/NAS2/tahir/descan_extension/AAAI_Github_Code_Descan/weights/Color_Encoder.h5'
# r50_512
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_512_retrain_20240919/10.pth' 
# r50_256
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256_20240912/50.pth'
# r34_512
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r34_512_20240923/70.pth'
# r50_fdl001_256
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256FDL_001_20240913/120.pth'
# r50_256_new
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256_20241004/50.pth' 

model, color_encoder = load_model_and_encoder(diffusion_weights_path,color_encoder_path)
beta = beta_schedule()
alpha, alpha_hat = prepare_alpha_schedules(beta)

test_folder = '/NAS2/tahir/descan_extension/data_set/Descan_dataset/Test'
test(test_folder, model, color_encoder, alpha, alpha_hat, 20) 

