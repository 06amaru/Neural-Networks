import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from PIL import Image
import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 2

img_t = transform.Compose([transform.ToTensor()])


train_set = torchvision.datasets.ImageFolder('dataset/train', transform=img_t)
val_set = torchvision.datasets.ImageFolder('dataset/val', transform=img_t)

print(train_set)

class Low_High_Dataset(Dataset):
    def __init__(self, path, tfms):
        self.path = Path(path)
        self.tfms = tfms
        self.len = len(fnmatch.filter(os.listdir(self.path / "high_res"), '*.png'))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        high_res_path = self.path / "high_res" / f"{idx}.png"
        low_res_path = self.path / "low_res" / f"{idx}.png"
        if self.tfms is not None:
            return self.tfms(np.array(Image.open(low_res_path))[...,:-1]), self.tfms(np.array(Image.open(high_res_path))[...,:-1])
        
train_set = Low_High_Dataset(Path("dataset/train"), img_t)
val_set = Low_High_Dataset(Path("dataset/val"), img_t)

train_loader = DataLoader(train_set, batch_size=2)
valid_loader = DataLoader(val_set, batch_size=2)

l, h = next(iter(train_loader))
l.shape, h.shape

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=0)
    self.pool1 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0)
    self.pool2 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.pool3 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.pool4 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
    self.pool5 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
    self.pool6 = nn.MaxPool2d(2, stride = 2, return_indices=True)
    self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)

  def forward(self, image):
    out1 = F.sigmoid(self.conv1(image))
    out1p, ind1 = self.pool1(out1)
    out2 = F.sigmoid(self.conv2(out1p))
    out2p, ind2 = self.pool2(out2)
    out3 = F.sigmoid(self.conv3(out2p))
    out3p, ind3 = self.pool3(out3)
    out4 = F.sigmoid(self.conv4(out3p))
    out4p, ind4 = self.pool4(out4)
    out5 = F.sigmoid(self.conv5(out4p))
    out5p, ind5 = self.pool5(out5)
    out6 = F.sigmoid(self.conv6(out5p))
    out6p, ind6 = self.pool6(out6)
    out7 = self.conv7(out6p)
    z = out7
    return z, out1, ind1, out2, ind2, out3, ind3, out4, ind4, out5, ind5, out6, ind6

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    self.convTran1 = nn.ConvTranspose2d(in_channels=512,out_channels=256, kernel_size=1, stride=1, padding=0)
    self.poolT1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran2 = nn.ConvTranspose2d(in_channels=256*2,out_channels=128, kernel_size=3, stride=1, padding=0)
    self.poolT2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran3 = nn.ConvTranspose2d(in_channels=128*2,out_channels=128, kernel_size=3, stride=1, padding=0)
    self.poolT3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran4 = nn.ConvTranspose2d(in_channels=128*2,out_channels=128, kernel_size=3, stride=1, padding=0)
    self.poolT4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran5 = nn.ConvTranspose2d(in_channels=128*2,out_channels=128, kernel_size=3, stride=1, padding=0)
    self.poolT5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran6 = nn.ConvTranspose2d(in_channels=128*2,out_channels=128, kernel_size=5, stride=1, padding=0)
    self.poolT6 = nn.MaxUnpool2d(kernel_size=2, stride=2)
    self.convTran7 = nn.ConvTranspose2d(in_channels=128*2,out_channels=3, kernel_size=7, stride=1, padding=0)
    
    
  def forward(self, latent, out1, ind1, out2, ind2, out3, ind3, out4, ind4, out5, ind5, out6, ind6):
    out_1 = self.convTran1(latent)
    out_1p = self.poolT1(out_1, ind6, output_size= out6.size())
    out_1p = torch.cat([out_1p, out6], 1)
    out_2 = F.sigmoid(self.convTran2(out_1p))
    out_2p = self.poolT2(out_2, ind5, output_size= out5.size())
    out_2p = torch.cat([out_2p, out5], 1)
    out_3 = F.sigmoid(self.convTran3(out_2p))
    out_3p = self.poolT3(out_3, ind4, output_size=out4.size())
    out_3p = torch.cat([out_3p, out4], 1)
    out_4 = F.sigmoid(self.convTran4(out_3p))
    out_4p = self.poolT4(out_4, ind3)
    out_4p = torch.cat([out_4p, out3], 1)
    out_5 = F.sigmoid(self.convTran5(out_4p))
    out_5p = self.poolT5(out_5, ind2, output_size= out2.size())
    out_5p = torch.cat([out_5p, out2], 1)
    out_6 = F.sigmoid(self.convTran6(out_5p))
    out_6p = self.poolT6(out_6, ind1, output_size= out1.size())
    out_6p = torch.cat([out_6p, out1], 1)
    out_7 = F.sigmoid(self.convTran7(out_6p))
    
    return out_7
  
class Autoencoder(nn.Module):
   def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

   def forward(self, x):
        latent, out1, ind1, out2, ind2, out3, ind3, out4, ind4, out5, ind5, out6, ind6 = self.encoder(x)
        x_recon = self.decoder(latent, out1, ind1, out2, ind2, out3, ind3, out4, ind4, out5, ind5, out6, ind6)
        return  x_recon
   
def train(model, train_loader, val_loader, Epochs, loss_fn):
    train_loss_avg = []
    val_loss_avg = []
    for epoch in tqdm(range(Epochs)):
        train_loss_avg.append(0)
        num_batches = 0
        for low_res, high_res in train_loader:
            high_res, low_res = high_res.cuda(), low_res.cuda()
            predicted_high_res = model(low_res)
            loss = loss_fn(predicted_high_res, high_res)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
          
            train_loss_avg[-1] += loss.item()
            num_batches += 1
        
        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, Epochs, train_loss_avg[-1]))

        val_loss_avg.append(0)
        num_batches=0
        for low_res, high_res in val_loader:
            with torch.no_grad():
                high_res, low_res = high_res.cuda(), low_res.cuda()
                predicted_high_res = model(low_res)            
                
                loss = loss_fn(predicted_high_res, high_res)
                val_loss_avg[-1] += loss.item()
                num_batches += 1
        val_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction validation error: %f' % (epoch+1, Epochs, val_loss_avg[-1]))
        
        
        for low_res, _ in val_loader:
          with torch.no_grad(): 
              high_res, low_res = high_res.cuda(), low_res.cuda()
              predicted_high_res = autoencoder(low_res)
              im = transform.ToPILImage()(low_res[0]).convert("RGB")  
              display(im)
              imt = transform.ToPILImage()(predicted_high_res[0]).convert("RGB")
              display(imt)
          break
        
    return train_loss_avg, val_loss_avg

epochs = 25
learning_rate = 0.0001
autoencoder = Autoencoder()
autoencoder.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

loss_result, loss_val = train(model=autoencoder, 
                              train_loader=train_loader,
                              val_loader=valid_loader, 
                              Epochs=epochs, loss_fn=loss_fn)

fig = plt.figure()
plt.plot(loss_result)
plt.plot(loss_val)
plt.xlabel('Epochs')
plt.ylabel('Reconstruction error')
plt.show()

for low_res, high_res in valid_loader:
    with torch.no_grad(): 
        high_res, low_res = high_res.cuda(), low_res.cuda()
        predicted_high_res = autoencoder(low_res)
        im = transform.ToPILImage()(low_res[0]).convert("RGB")  
        display(im)
        imt = transform.ToPILImage()(predicted_high_res[0]).convert("RGB")
        display(imt)
