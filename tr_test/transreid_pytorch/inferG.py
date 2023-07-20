import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import modifier         
import argparse
from config import cfg
from datasets.dataload import test_gan
import logging

def show_data(dataloader, modelG, num=0):
    device = torch.device("cuda")
    i = iter(dataloader)
    for _ in range(num):
        real_batch = next(i)
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    print(real_batch[0].to(device)[0].shape)
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[0], padding=2, normalize=True).cpu(),(1,2,0)))

    output = modelG(real_batch[0].to(device)[:64])
    #output = torch.clamp(output, min=-1, max=1)

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(vutils.make_grid(output[0], padding=2, normalize=True).cpu(), (1,2,0)))
    plt.show()

# set seed
seed = random.randint(1, 20) 

# cuda
device = torch.device("cuda")

# parser
parser = argparse.ArgumentParser(description="Test the Gan")
parser.add_argument(
    "--config_file",
    default="/storage/sjhwang/TransReID-SSL/transreid_pytorch/configs/market/vit_small_ics.yml", 
    help="path to config file", 
    type=str)
parser.add_argument(
    "--model_p", 
    default="/storage/sjhwang/tr_test/transreid_pytorch/log/transreid/market/again/Hello", 
    help="path to config file", 
    type=str)
parser.add_argument(
    "--data_p",
    default="/storage/sjhwang/TransReID-SSL/data/market1501",
    help="path to dataset",
    type=str)
parser.add_argument(
    "opts", 
    help="Modify config options using the command-line", 
    default=None,
    nargs=argparse.REMAINDER)
args = parser.parse_args()

# set config file
if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

# lambda
lambda_l1 = 0.3

# epoch
epoch = 50

# modifier
modelG = modifier.Style_net()
modelG.load_state_dict(torch.load(args.model_p + '/transformerG_0.1_{}.pth'.format(epoch)))
modelG.to(device)
modelG.eval()

# log
te = 'gan_infer_test_seed_{}_transformerG_{}_{}'.format(seed,lambda_l1,epoch)
logger = logging.getLogger(te)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)s] %(message)s')

# save the log
if not os.path.isdir('./log/infer'):
    os.mkdir('./log/infer')
file_handler = logging.FileHandler('./log/infer/' + te + '.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# dataloader
train_dataloader, gallery_dataloader, query_dataloader, num_classes, cam_num, view_num = test_gan(cfg)

# visualization the data
show_data(train_dataloader, modelG, seed)
#show_data(gallery_dataloader, modelG, seed)
#show_data(query_dataloader, modelG, seed)

# test the quality
#is_good = input("test the quality ")
#logger.info('test the quality ' + is_good)


