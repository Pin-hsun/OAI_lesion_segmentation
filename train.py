import time, os
from torch.utils.data import DataLoader
from utils.args import args_train, merge_args
import torch.nn as nn
from engine.lightning_classification import LitClassification
from dotenv import load_dotenv
load_dotenv('.env')
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import random


def train(net, args, train_set, eval_set, loss_function, metrics):
    # Data Loader
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, drop_last=False)
    eval_loader = DataLoader(eval_set, batch_size=args['batch_size'], shuffle=False, num_workers=4, drop_last=False)

    # Load the model to GPU
    if args['legacy']:
        if not args['cpu']:
            net = net.cuda()
        # Run the models in parallel
        if args['parallel']:
            net = nn.DataParallel(net)

    # Sometimes with freeze a part of the model to reduce the number of parameters
    net.par_freeze = []

    # Define Final Model
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
            mode='min'
        )
        # Ray_callbacks = TuneReportCallback(metrics, on="validation_end")
        # we can use loggers (from TensorBoard) to monitor the progress of training
        tb_logger = pl_loggers.TensorBoardLogger('logs/' + args['prj'] + '/')
        trainer = pl.Trainer(gpus=1, accelerator='ddp',
                             max_epochs=200, progress_bar_refresh_rate=10, logger=tb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(ln_classification, train_loader, eval_loader)


if __name__ == "__main__":
    from loaders.loader_annotation import LoaderImorphics as Loader
    from models.unet import UNet_clean
    from utils.metrics_segmentation import SegmentationCrossEntropyLoss, SegmentationDiceCoefficient

    # Training Arguments
    parser = args_train()
    args = dict(vars(parser.parse_args()))
    args['dir_checkpoint'] = os.environ.get('CHECKPOINTS')

    # Datasets
    # Dataset Arguments
    args_d = {'mask_name': 'pin',
              # 'data_path': os.getenv("HOME") + os.environ.get('DATASET'),
              'data_path': os.environ.get('DATASET'),
              'mask_used': [['png']],  #[[1], [2, 3]],  # ['femur'], ['tibia'],
              'scale': 1,
              'interval': 1,
              'thickness': 0,
              'method': 'automatic'}

    args = merge_args(args, args_d)

    # Splitting Subjects
    def imorphics_split():
        DIR = args['data_path'] + '/img'
        cnt = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        random.seed(42)
        eval_00 = random.sample(range(0, cnt), 100)
        train_00 = list(set(range(0, cnt)) - set(eval_00))
        # train_00 = list(range(0, cnt-100))
        # eval_00 = list(range(cnt-100, cnt))
        train_01 = list(range(10+88, 71+88))
        eval_01 = list(range(1+88, 10+88)) + list(range(71+88, 89+88))
        return train_00, eval_00, train_01, eval_01

    train_00, eval_00, train_01, eval_01 = imorphics_split()

    # Dataloader
    train_set = Loader(args, subjects_list=train_00, mode='train')
    eval_set = Loader(args, subjects_list=eval_00, mode='eval')
    print('Length of training set')
    print(len(train_set))
    print('Length of Validation set')
    print(len(eval_set))

    # Model, Loss Function, Metrics
    # old API
    # net = UNet_clean(output_ch=len(args_d['mask_used']) + 1, backbone=args['backbone'], depth=args['depth'])
    # new API
    from segmentation_models_pytorch.unet import Unet
    net = Unet(encoder_name='vgg11', classes=len(args_d['mask_used']) + 1, activation=None, encoder_depth=5)
    loss_function = SegmentationCrossEntropyLoss()
    metrics = SegmentationDiceCoefficient()

    # Start Training
    train(net, args, train_set, eval_set, loss_function, metrics)

# Usage in command line:
# CUDA_VISIBLE_DEVICES=2 python train.py -b 16 --bu 64 --lr 0.001 --legacy
