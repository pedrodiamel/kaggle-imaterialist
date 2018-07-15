import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.backends.cudnn as cudnn

from pytvision.transforms import transforms as mtrans
from pytvision import visualization as view

sys.path.append('../')
from torchlib.datasets.factory  import FactoryDataset 
from torchlib.datasets import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

root= '~/.datasets'
pathname = os.path.expanduser(root)
name_dataset=FactoryDataset.afew

dataloader = Dataset(
    data=FactoryDataset.factory(pathname=pathname, 
        name=name_dataset, 
        subset=FactoryDataset.validation, 
        download=True ),
    num_channels=3,
    transform=transforms.Compose([
        mtrans.ToResize( (64, 64), resize_mode='square', padding_mode=cv2.BORDER_REFLECT101 ),
        mtrans.ToTensor(),
        mtrans.ToNormalization(),
        ])
    )

print(dataloader.labels)
print(dataloader.classes)
#print(dataloader.data.classes)
#print(dataloader.data.class_to_idx)

print( dataloader[0]['label'].shape, dataloader[0]['image'].shape )
print( len(dataloader) )

plt.figure( figsize=(16,16))
view.visualizatedataset(dataloader, num=100, imsize=(64,64,3) )
plt.axis('off')
plt.ioff()
plt.show() 

