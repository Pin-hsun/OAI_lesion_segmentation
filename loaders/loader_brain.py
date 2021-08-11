import os
import cv2
import numpy as np
from PIL import Image
import scipy.io
import torch
from torch.utils import data
import random
#from cStringIO import StringIO
from io import StringIO
from io import BytesIO
import glob
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


class DataBrain(Dataset):
    def __init__(self, source):
        self.img_list = sorted(glob.glob(source + '/original/*'))
        self.edges_list = sorted(glob.glob(source + 'edges/*'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_list[index]))
        gt = np.array(Image.open(self.edges_list[index]))

        if len(img.shape) == 2:
            img = np.concatenate([np.expand_dims(img, 0)]*3, 0)

        img = img / img.max()
        img = img.astype(np.float32)

        gt = ((gt[:, :, :].sum(2)) > 0) / 1
        gt = np.expand_dims(gt, 0)
        gt = gt.astype(np.int64)

        return img, gt, '1'


#import bdcn
if __name__ == '__main__':
    ds = DataBrain(source='/media/ghc/GHc_data1/Dataset/heddata/brain0708/patches/test/')
    img, gt, _ = ds.__getitem__(100)
    net = torch.load('checkpoints/15.pth')
    a = net(torch.from_numpy(img).unsqueeze(0).cuda())
    a = [x.detach().cpu() for x in a]
    imagesc(img[0, ::])
    imagesc(gt[0, ::])
    prob = nn.Softmax(dim=1)(a[0])
    imagesc(a[0][0, 1, ::])
    imagesc(torch.argmax(a[0][0, ::], 0))

    x = np.array(Image.open(sorted(glob.glob('/media/ghc/GHc_data1/Dataset/heddata/brain0708/original/*'))[20]))
    y = np.array(Image.open(sorted(glob.glob('/media/ghc/GHc_data1/Dataset/heddata/brain0708/scale0708/*'))[20]))
    x = x / x.max()
    x = np.concatenate([np.expand_dims(x, 0)] * 3, 0)
    x = x.astype(np.float32)
    a = net(torch.from_numpy(x).unsqueeze(0).cuda())
    a = [x.detach().cpu() for x in a]
    #imagesc(a[0][0, 1, ::])
    prob = nn.Softmax(dim=1)(a[0])
    imagesc(x)
    imagesc((prob[0, 1, ::] > 0.7) / 1)
    #    imagesc(torch.argmax(a[-1][0, ::], 0).detach().cpu())