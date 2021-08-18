import torch
import os
from utils.imagesc import imagesc
from dotenv import load_dotenv
load_dotenv('.env')
from loaders.loader_imorphics import LoaderImorphics as Loader


args_d = {'mask_name': 'bone_resize_B_crop_00',
          'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
          'mask_used': [['femur'], ['tibia']],  # [[1], [2, 3]],  # ['femur'], ['tibia'],
          'scale': 0.5,
          'interval': 1,
          'thickness': 0,
          'method': 'automatic'}


def imorphics_split():
    train_00 = list(range(10, 71))
    eval_00 = list(range(1, 10)) + list(range(71, 89))
    train_01 = list(range(10 + 88, 71 + 88))
    eval_01 = list(range(1 + 88, 10 + 88)) + list(range(71 + 88, 89 + 88))
    return train_00, eval_00, train_01, eval_01


def simple_test():
    """
    A simple test function to prnt out
    """
    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # Dataloader
    train_set = Loader(args_d, subjects_list=train_00)
    eval_set = Loader(args_d, subjects_list=eval_00)

    # Loading Data
    x, y, id = eval_set.__getitem__(50)
    print(id)
    print('shape of input')
    print(x.shape)
    print('shape of label')
    print(y.shape)

    #
    model = torch.load('checkpoints/190.pth')
    out, = model(x.unsqueeze(0).cuda())
    print(out.shape)
    segmentation = torch.argmax(out, 1).detach().cpu()
    print(segmentation.shape)

    to_print = torch.cat([x/x.max() for x in [x[0, ::], segmentation[0, ::]]], 1)
    imagesc(to_print, show=False, save='segmentation.png')


if __name__ == '__main__':
    simple_test()