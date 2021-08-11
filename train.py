import time, os
from optparse import OptionParser
from torch.utils.data import DataLoader
import torch.nn as nn
from engine.lightning_classification import LitClassification
from dotenv import load_dotenv
load_dotenv('.env')
#from pytorch_lightning.callbacks import ModelCheckpoint
#import pytorch_lightning as pl
#from pytorch_lightning import loggers as pl_loggers


def merge_args(args, args_d):
    args.update(args_d)
    return args


def args_train():
    # Training Parameters
    parser = OptionParser()
    # Name of the Project
    parser.add_option('--prj', dest='prj', default='dess_segmentation',
                      type=str, help='name of the project')
    # How many epoch to run?
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    # Batch size per mini-batch
    parser.add_option('-b', '--batch-size', dest='batch_size', default=16,
                      type='int', help='batch size')
    # Sometimes we update the model after several minibatch
    parser.add_option('--bu', '--batch-update', dest='batch_update', default=64,
                      type='int', help='batch to update')
    # Learning rate: How fast we want to train the model?
    parser.add_option('--lr', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    # Use GPU?
    parser.add_option('-c', '--cpu', action='store_true', dest='cpu', default=False, help='only use cpu'),

    parser.add_option("--sv", action="store_true", dest='save_cp', default=False, help='save model parameters'),
    parser.add_option('-p', '--pred', action='store_true', dest='pred', default=False, help='only evaluate model')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load saved model parameters')
    parser.add_option('--par', dest='parallel', action="store_true", help='run in multiple gpus')
    parser.add_option('-w', '--weight-decay', dest='weight_decay', default=0.0005,
                      type='float', help='weight decay')
    parser.add_option('--ini', '--ini-file', dest='ini_file', default='latest',
                      type=str, help='name of the ini file')
    parser.add_option('--legacy', action='store_true', dest='legacy', default=True, help='legacy pytorch')

    # Model Parameters
    parser.add_option('--backbone', dest='backbone', default='vgg11',
                      type=str, help='backbone of the UNet')
    parser.add_option('--depth', dest='depth', default=5,
                      type=int, help='Number of layers of UNet')

    # Misc
    parser.add_option('--mode', type=str, default='dummy')
    parser.add_option('--port', type=str, default='dummy')

    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":

    args = vars(args_train())
    args['dir_checkpoint'] = os.getenv("HOME") + os.environ.get('CHECKPOINTS')

    args_d = {'mask_name': 'bone_resize_B_crop_00',
              'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
              'mask_used': [['femur', 'tibia'], [1], [2, 3]],  # ['femur'], ['tibia'],
              'scale': 0.5,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}

    args = merge_args(args, args_d)

    """ split range"""
    def imorphics_split():
        train_00 = list(range(10, 71))
        eval_00 = list(range(1, 10)) + list(range(71, 89))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01

    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # datasets
    from loaders.loader_imorphics import LoaderImorphics
    train_set = LoaderImorphics(args_d, subjects_list=train_00)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=16, drop_last=False)
    eval_set = LoaderImorphics(args_d, subjects_list=eval_00)
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'], shuffle=False, num_workers=16, drop_last=False)
    print(len(train_set))
    print(len(eval_set))

    """ Imports """
    from models.unet import UNet_clean
    net = UNet_clean(output_ch=len(args_d['mask_used']) + 1, backbone=args['backbone'], depth=args['depth'])
    from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient
    loss_function = SegmentationCrossEntropyLoss()
    metrics = SegmentationDiceCoefficient()

    """ cuda """
    # Load the models to GPU
    if args['legacy']:
        if not args['cpu']:
            net = net.cuda()
        # Run the models in parallel
        if args['parallel']:
            net = nn.DataParallel(net)
    # Sometimes with freeze a part of the model to reduce the number of parameters
    net.par_freeze = []

    """ Lightning """
    ln_classification = LitClassification(args=args,
                                          train_loader=train_loader,
                                          eval_loader=eval_loader,
                                          net=net,
                                          loss_function=loss_function,
                                          metrics=metrics)
    if args['legacy']:
        # Use pytorch without lightning
        ln_classification.overall_loop()
    else:
        # Use pytorch lightning for training, you can ignore it
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints/' + args['prj'] + '/',
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )
        # we can use loggers (from TensorBoard) to monitor the progress of training
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args['prj'] + '/')
        trainer = pl.Trainer(gpus=4, accelerator='ddp',
                             max_epochs=100, progress_bar_refresh_rate=20, logger=tb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(ln_classification, train_loader, eval_loader)


# CUDA_VISIBLE_DEVICES=0 python train.py -b 16 --bu 64 --lr 0.01

