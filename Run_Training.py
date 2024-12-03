import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from Trainer_descan import *  # Assuming necessary diffusion-related definitions are here
# from Conditional_Reverse_Function import *  # Assuming this contains conditional reverse functions
from Conditional_Reverse_Function_CBAM import *  # Assuming this contains conditional reverse functions
import warnings

# Configuration Section: Paths and Parameters
# --------------------------------------------
# Paths
log_name = './logs_train/Train_Logger_DD256_R34_CBAM_2_20241104.txt'
dataset_dir = '/home/tahir/workspace/descan_extension/data_set/Descan_dataset/Train'
# dataset_dir = '/home/tahir/workspace/descan_extension/data_set/Descan_dataset/Valid'

color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights/Color_Encoder.h5'
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_256/30.pth'
# r50 fdl 001
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_512_retrain_20240919/10.pth'
# color_encoder_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256FDL_001_20240913/120.pth'
valid_path = '/home/tahir/workspace/descan_extension/data_set/Descan_dataset/Valid'

gpu_id = [0, 1, 2, 3]
# gpu_id = [0]
# Training Parameters
batch_size = 16
steps = 2000
min_beta = 0.0001
max_beta = 0.01
distort_threshold = 0.25
num_epochs = 100000
# num_epochs = 1
learning_rate = 0.0001

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Code that generates warnings goes here


# Logging Setup
with open(log_name, 'w') as log_file:
    log_file.write('Starting the logger\n')

# Data Loading and Transformation Setup
custom_set = Custom_Loader(dataset_dir, min_beta, max_beta, steps, distort_threshold, schedule='LINEAR')
train_set = DataLoader(custom_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

# Model, Loss, and Optimizer Setup
# ---------------------------------
color_encoder = models.resnet34(pretrained=False, progress=True)
color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False) #6 or 96
color_encoder.to(device)
#----------------------------------
class ModifiedResNet50(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNet50, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Keep all layers except the final fc
        
        # Define the new layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 6)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to ensure outputs are between 0 and 1
        
    def forward(self, x):
        x = self.base_model(x)  # Forward pass through ResNet50 backbone
        x = torch.flatten(x, 1)  # Flatten the output from the base model
        x = self.fc1(x)  # First linear layer (2048 -> 512)
        x = nn.ReLU()(x)  # Apply ReLU activation
        x = self.fc2(x)  # Second linear layer (512 -> 6)
        x = self.sigmoid(x)  # Sigmoid activation to bound the output between 0 and 1
        return x
    
# color_encoder = ModifiedResNet50(models.resnet50(pretrained=False, progress=True)) 
# color_encoder = color_encoder.to(device) 

checkpoint = torch.load(color_encoder_path, map_location=device)
# checkpoint = torch.load(color_encoder_path, map_location="cuda:0")
# color_encoder.load_state_dict(checkpoint['model_state_dict']) #.pth file
color_encoder.load_state_dict(checkpoint) #.h5 file

unet = Unet(dim=64, channels=6, out_dim=3, dim_mults=(1, 2, 4, 8, 8)).to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
unet.apply(weights_init)

model = nn.DataParallel(unet, device_ids=gpu_id)

# ## to load pretrained model to resume training
# diffusion_weights_path = '//home/dircon/descan_extension/AAAI_Github_Code_Descan/weights/20240712/40.pth'
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_CBAM256_20240912/220.pth'
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_D256_20240929/230.pth' 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_D256_FDL_20240929/110.pth' 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_FDL_Tuner_20241016/90.pth' 
# diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD512_R34_Tuner_20241025/100.pth'
# model_checkpoint = torch.load(diffusion_weights_path, map_location=device)
# model.load_state_dict(model_checkpoint['model_state_dict'])
# ##
diffusion_weights_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights_DD256_R34_CBAM_20241104/50.pth'
model_checkpoint = torch.load(diffusion_weights_path, map_location=device)
model.load_state_dict(model_checkpoint['model_state_dict'])
##### CNN TUNERS
# for name, param in model.named_parameters(): 
#     if 'tuners' not in name: 
#         param.requires_grad = False    
#     else: 
#         param.requires_grad = True     
#####
##### CBAM
# for name, param in model.named_parameters(): 
#     if 'cbam' not in name: 
#         param.requires_grad = False    
#     else: 
#         param.requires_grad = True     
#####

Loss_obj = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Log Training Configuration
with open(log_name, 'a') as log_file:
    log_file.write(f'TRAINING PARAMETERS\nBatch Size: {batch_size}, Time Steps: {steps}, Beta Range: ({min_beta}, {max_beta})\n')

# Training
epoch_done = 0
model_trained = train_model(model, color_encoder, optimizer, Loss_obj, epoch_done, valid_path, train_set, custom_set, log_name=log_name, num_epochs=num_epochs)

# Note: Ensure that the `train_model` function is defined with appropriate parameters including `device` and `log_name`.
# srun --gres=gpu:8 --cpus-per-gpu=8 --time 4-00:00:00 --mem-per-gpu=24G --partition batch --pty bash