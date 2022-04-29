import os, glob, torch, time
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
import tifffile as tiff
from dotenv import load_dotenv
load_dotenv('.env')

def split_img(mask_file, id):
    ori_path = os.environ.get('DATASET') + mask_file
    img_path = os.environ.get('DATASET') + 'img'
    mask_path = os.environ.get('DATASET') + 'mask/png'
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    im = Image.open(glob.glob(os.path.join(ori_path, id) + '.*')[0])
    for i, page in enumerate(ImageSequence.Iterator(im)):
        ori = np.array(page)
        img = ori[:, :384, 0]
        mask = ori[:, 384:, 0]
        Image.fromarray(img).save(img_path + f'/{id}_{i}.png')
        Image.fromarray(mask).save(mask_path + f'/{id}_{i}.png')

if __name__ == '__main__':
    mask_file = 'ziyi'
    id_ls = [x.split('/')[-1].split('.')[0] for x in glob.glob(os.path.join(os.environ.get('DATASET'), mask_file)+'/*')]
    for id in id_ls:
        split_img(mask_file, id)
