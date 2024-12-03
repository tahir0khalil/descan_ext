

import numpy as np
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
import natsort
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter,ImageOps,ImageEnhance
import os
import os.path
from pathlib import Path
from os import listdir
from os.path import join
from matplotlib import cm 
import sys
import random
from scipy.io import loadmat

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader ,random_split
from torch.nn import functional as F
import torchvision.transforms.functional as tf
import torch.optim as optim
import tqdm



# log_name = './Training_Loggger_text.txt'
log_name = './Encoder_logs_20241004_r50_256.txt'
print('Starting the logger for 256 sized global color correction module',  file=open(log_name, 'a'))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



data_dir_inp ='/home/tahir/workspace/descan_extension/data_set/Descan_dataset/Train'



class Custom_Loader(Dataset):
    def __init__(self, parent_dir):
        
        self.dir_clean, self.dir_scan = self.image_paths(parent_dir)
        self.Transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(256)
                                              ])

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"])

 
    def load_img(self, img_path):
        img = Image.open(img_path)
        return img 

    

    def image_paths(self, parent_dir):

        dir_clean = []
        dir_scan = []

        for x in  natsort.natsorted(listdir(join(parent_dir,'clean_512')),reverse=True):
            if os.path.isfile(join(parent_dir,'clean_512',x)) and os.path.isfile(join(parent_dir,'scan_512',x)):
                dir_clean.append(join(parent_dir,'clean_512', x))
                dir_scan.append(join(parent_dir, 'scan_512', x))
        return dir_clean, dir_scan
    
    def __len__(self):
        return len(self.dir_clean)


    def __getitem__(self, index):
        np.random.seed()
        img_clean = self.load_img(self.dir_clean[index])
        img_scan = self.load_img(self.dir_scan[index])

        hflip = random.randint(0,1)

        mirr  = random.random() > 0.5

        if hflip:
            img_clean = ImageOps.flip(img_clean)
            img_scan = ImageOps.flip(img_scan)

        if mirr:
            img_clean = ImageOps.mirror(img_clean)
            img_scan = ImageOps.mirror(img_scan)
            
        img_clean = self.Transforms(np.array(img_clean))
        img_scan = self.Transforms(np.array(img_scan))
        
        r = img_clean[0,:,:]
        g = img_clean[1,:,:]
        b = img_clean[2,:,:]

        
        r_mean, g_mean, b_mean = torch.mean(r), torch.mean(g), torch.mean(b)
        r_std, g_std, b_std = torch.std(r), torch.std(g), torch.std(b)

        clean_dist =  torch.tensor([r_mean, g_mean, b_mean, r_std, g_std, b_std])

        return img_scan, clean_dist


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.axis('off')

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=None)

    
    
class PSNR:
# """Peak Signal to Noise Ratio
# img1 and img2 have range [0, 1]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

psnr = PSNR()


batch= 32

image_datasets = Custom_Loader(data_dir_inp)
print(len(image_datasets))

dataloaders =DataLoader(image_datasets,
                                batch_size=batch,
                                shuffle=True,drop_last=True,num_workers=0)



print('TRAINING PARAMETERS',  file=open(log_name, 'a'))
print('batch '+str(batch),  file=open(log_name, 'a'))




# model = torchvision.models.resnet34(pretrained = False, progress = True)
# model.fc = nn.Linear(in_features=512, out_features=6, bias=False)


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
    
    
model = ModifiedResNet50(models.resnet50(pretrained=False, progress=True)) 
model = model.to(device)
# model.to(device)

checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256_2_20240912/50.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

Transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(512)
                                        ])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.0003,weight_decay = 0.00001)



def mean_std_image(image):
    img_data = image.copy()
    # Split the image into R, G, B channels
    r, g, b = img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]
    # Calculate the average and standard deviation for each channel
    r_avg, g_avg, b_avg = np.mean(r), np.mean(g), np.mean(b)
    r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)


    mean = r_avg, g_avg, b_avg
    std = r_std, g_std, b_std
    return  mean,std


def color_shift_function(scan_image ,mean_clean,std_clean,mean_scan,std_scan):
    ### scan_image np array.
    pixels_b = scan_image.copy()
    r_avg, g_avg, b_avg =  mean_clean
    r_std, g_std, b_std = std_clean

    r_avg_b, g_avg_b, b_avg_b = mean_scan
    r_std_b, g_std_b, b_std_b = std_scan
    
    r = pixels_b[:,:,0]
    g = pixels_b[:,:,1]
    b = pixels_b[:,:,2]

    # print(r.shape)
    n_r,n_g,n_b = (r-r_avg_b)/r_std_b , (g-g_avg_b)/g_std_b, (b-b_avg_b)/b_std_b
    pixels_b[:, :,0] = n_r*r_std+r_avg
    pixels_b[:, :,1] = n_g*g_std+g_avg
    pixels_b[:, :,2] = n_b*b_std+b_avg
    # print(np.max(pixels_b))
    pixels_b[:,:,0] = pixels_b[:,:,0]/np.max(pixels_b[:,:,0])
    pixels_b[:,:,1] = pixels_b[:,:,1]/np.max(pixels_b[:,:,1])
    pixels_b[:,:,2] = pixels_b[:,:,2]/np.max(pixels_b[:,:,2])
    

    return pixels_b


def validate(test_folder, model_function, epoch):
    newpath =  './resnet50_256_20241004/'+str(epoch)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    total_PSNR = []
    print('Validation '+str(epoch),  file=open(log_name, 'a'))
    for x in  natsort.natsorted(listdir(join(test_folder, 'scan')), reverse=True):
        img_scan = Transforms(Image.open(join(test_folder, 'scan', x)))
        img_clean =  Transforms(Image.open(join(test_folder, 'clean', x)))

        output = model_function(img_scan[None, ...].to(device))
        [r_mean, g_mean, b_mean, r_std, g_std, b_std] = output[0].cpu().numpy() # * 255

        mean_pred = r_mean, g_mean, b_mean
        std_pred = r_std, g_std, b_std

        image_scan = Image.open(join(test_folder, 'scan', x))
        image_scan = image_scan.resize((512, 512)) 
        image_scan = np.array(image_scan.convert('RGB')).astype(np.float32) / 255
        mean_scan, std_scan = mean_std_image(image_scan)

        image_clean = Image.open(join(test_folder, 'clean', x))
        image_clean = image_clean.resize((512, 512)) 
        image_clean = np.array(image_clean.convert('RGB')).astype(np.float32) / 255
        mean_clean, std_clean = mean_std_image(image_clean)   

        pred_shift = color_shift_function(image_scan.copy(), mean_pred, std_pred, mean_scan, std_scan)
        clean_shift = color_shift_function(image_scan.copy(), mean_clean, std_clean, mean_scan, std_scan)
 
        pred_shift = Transforms(pred_shift)
        clean_shift = Transforms(clean_shift)

        print('Name: ', x, '  ', psnr(pred_shift, clean_shift))
        print('Name: '+x, '  PSNR '+str(psnr(pred_shift, clean_shift).item()),  file=open(log_name, 'a'))
        total_PSNR.append(psnr(pred_shift, clean_shift).item())

        torchvision.utils.save_image(pred_shift, newpath+'/'+x+'_'+str(epoch+1)+".png")

    avg_psnr = np.mean(np.array(total_PSNR))
    print('Average Validation PSNR', avg_psnr)

    print('Average Validation PSNR', avg_psnr,  file=open(log_name, 'a'))


def train_model(model, test_folder, criterion, optimizer, num_epochs=3):
    loss_train = []
    Epochs = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        i = 0

        for img_in, clean_dist in dataloaders:
            inputs = img_in.to(device)
            clean_dist = clean_dist.to(device)
            i += 1
            outputs = model(inputs)
            loss = criterion(outputs, clean_dist)
            print(f'\r {i}  loss {loss.item()}', end='')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * inputs.size(0)

        epoch_loss = running_loss / len(image_datasets)
        Epochs.append(epoch)
        loss_train.append(epoch_loss)

        print(f'EPOCH {epoch}  Epoch Loss: {epoch_loss.item()}  Learning Rate: {optimizer.param_groups[0]["lr"]}', file=open(log_name, 'a'))
        print(f'Epoch Loss: {epoch_loss.item()}')

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                validate(test_folder, model, epoch)
            print('Saved history')

        if (epoch + 1) % 10 == 0:
            newpath = './weights/Color_encoder_r50_256_20241004'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{newpath}/{epoch + 1}.pth')

    return model


test_path = '/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/valid_small'

model_trained = train_model(model,test_path, criterion, optimizer, num_epochs=150)



# torch.save(model.state_dict(), 'Color_encoder_128.h5')





