#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
#from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
matplotlib.rcParams['figure.facecolor'] = '#ffffff'


# In[18]:


# Set training dataset and validation dataset
train_ds = CIFAR100(root = 'E:/data/cifar100', train=True, download=False, transform=tt.ToTensor())
val_ds = CIFAR100(root = 'E:/data/cifar100', train=False, download=False, transform=tt.ToTensor())


# In[23]:


val_ds[0]


# In[3]:


ds = CIFAR100(root = 'E:/data/cifar100', download=False, transform=tt.ToTensor())


# In[4]:


num_classes = len(ds.classes)
image, label = ds[0]


# In[5]:


def show_example(img, label):
    print('Label: ', train_ds.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0)) # because imshow expects RGB color channel as 3rd dimension


# In[6]:


show_example(*train_ds[0])


# In[7]:


# Data transforms (normalization & data augmentation)
stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)) # ((mean for pixel intensity for red, mean for green, mean for blue), (SD for red, SD for green, SD for blue))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),  # padding first transforms 32x32 image into 40x40, then RandomCrop crops out 32x32 from the image. Reflect makes it look as if padding is a mirror.
                         tt.RandomHorizontalFlip(),
                         # tt.RandomVerticalFlip(),
                         # tt.RandomRotation,
                         # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                         # tt.RandomErasing(p = 0.5, scale=(0.03, 0.08)),
                         tt.GaussianBlur(5,sigma=0.18),
                         tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True),
                         tt.RandomErasing(p = 0.5, scale=(0.03, 0.08))])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


# In[25]:


# PyTorch datasets
train_ds = CIFAR100(root='E:/data/cifar100', download=False, train=True, transform=train_tfms)
valid_ds = CIFAR100(root='E:/data/cifar100', download=False, train=False, transform=valid_tfms)


# In[26]:


valid_ds[0]


# In[27]:


batch_size = 250
# PyTorch data loaders
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)


# In[29]:


valid_dl[0]


# In[10]:


#take a look at few sample images, post-denormalizing.
def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1))
        break


# In[11]:


#GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[12]:


device = get_default_device()
device


# In[13]:


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


# In[30]:


import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# In[15]:


#general functions and classes
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# In[16]:


#RESNET9
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers) # This is equal to nn.Sequential(listing out each item in layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 3x32x32
        self.conv1 = conv_block(in_channels, 64) # 64x32x32
        self.conv2 = conv_block(64, 128, pool=True) # 128x16x16
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) # 128x16x16
        
        self.conv3 = conv_block(128, 256, pool=True) # 256x8x8
        self.conv4 = conv_block(256, 512, pool=True) # 512x4x4
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) # 512x4x4
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), # 512x1x1
                                        nn.Flatten(),   # 3d image to 1d (i.e., shape = 512)
                                        nn.Dropout(0.1),  # to avoid overfitting, randomly select 20% of values that came out from previous layer and zero them out. Makes the model more generic.
                                        nn.Linear(512, num_classes)) # 10
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# In[17]:


model = to_device(ResNet9(3, num_classes), device)
model


# In[18]:


#TRAINNING
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache() # empty out the cache and use the entire GPU memory
    history = []
    
    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# In[19]:


#Pre-Training Accuracy
history = [evaluate(model, valid_dl)]
history


# In[20]:


#Actual Training
epochs = 25
max_lr = 0.012
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam


# In[27]:


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    t = "{0:02d}:{1:02d}:{2:02d}".format(h, m, s)
    print (t)
   #print("Execution Time: " + "{0:02d}:{1:02d}:{2:02d}".format(h, m, s))
    return t


# In[33]:


get_ipython().run_cell_magic('time', '', 'start = time.time()\nhistory += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, \n                                  grad_clip=grad_clip, \n                                  weight_decay=weight_decay, \n                                  opt_func=opt_func)\nend = time.time()\nttaken = exec_time(start, end)')


# In[34]:


train_time = ttaken


# In[35]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');


# In[36]:


plot_accuracies(history)


# In[37]:


plot_lrs(history)


# In[38]:


plot_losses(history)


# In[39]:


def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]


# In[40]:


img, label = valid_ds[0]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))


# In[41]:


img, label = valid_ds[-1]
plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
print('Label:', train_ds.classes[label], ', Predicted:', predict_image(img, model))


# In[42]:


history


# In[44]:


accuracies = [x['val_acc'] for x in history]


# In[45]:


accuracies


# In[ ]:




