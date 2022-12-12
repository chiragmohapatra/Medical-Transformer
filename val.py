import numpy as np
import torch
import torch.nn.functional as F
import argparse
import lib
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--data', type=str)
parser.add_argument('--modelname', default='off', type=str,
                    help='name of the model to load')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--save', action='store_true', help='store images or not')
args = parser.parse_args()

direc = args.direc
modelname = args.modelname
imgsize = args.imgsize
loaddirec = args.loaddirec


from utils import JointTransform2D, ImageToImage2D, Image2D
imgchant = 3
crop = None

tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
valloader = DataLoader(val_dataset, 1, shuffle=False)

device = torch.device("cuda")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size = imgsize, imgchan = imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size = imgsize, imgchan = imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size = imgsize, imgchan = imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size = imgsize, imgchan = imgchant)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

model.load_state_dict(torch.load(loaddirec))
model.eval()

iou = 0
dice = 0
num = 0

for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
    # print(batch_idx)
    if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
    else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

    X_batch = Variable(X_batch.to(device='cuda'))
    y_batch = Variable(y_batch.to(device='cuda'))

    y_out = model(X_batch)

    tmp2 = y_batch.detach().cpu().numpy()
    tmp = y_out.detach().cpu().numpy()
    
    iou += iou_score(tmp,tmp2)
    dice += dice_coef(tmp,tmp2)
    num += 1
    
    tmp[tmp>=0.5] = 1
    tmp[tmp<0.5] = 0
    tmp2[tmp2>0] = 1
    tmp2[tmp2<=0] = 0
    tmp2 = tmp2.astype(int)
    tmp = tmp.astype(int)

    # print(np.unique(tmp2))
    yHaT = tmp
    yval = tmp2
    
    del X_batch, y_batch,tmp,tmp2, y_out

    yHaT[yHaT==1] =255
    yval[yval==1] =255
    
    if save:
      fulldir = direc+"/"

      if not os.path.isdir(fulldir):

          os.makedirs(fulldir)

      cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])
      
print('Dice Score :', dice/num)
print('mIOU :', iou/num)
