from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets, models
from torch import nn

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model

from losses import SupConLoss, BTLoss
from APViT import IRes_50

# for kdef
from torch.utils import data
from torchvision import transforms, utils
import os
from PIL import Image
import torch

import pickle

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='kdef',
                        choices=['cifar10', 'cifar100', 'path','kdef'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='efficient of loss_BT in loss')
    parser.add_argument('--beta', type=float, default='0.9',
                        help='efficient of loss_BT in loss')
    parser.add_argument('--gamma', type=float, default='0.5',
                        help='efficient of loss_BT in loss') 
    parser.add_argument('--special', type=str, default='None',
                        help='illustration of special handling')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='ratio of loss*3')
    
    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_{}_models'.format(opt.dataset,opt.method)
    opt.tb_path = './save/SupCon/{}_{}_tensorboard'.format(opt.dataset,opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '2{}_{}_{}_bsz_{}_epoch_{}_alpha_{}_beta_{}_gamma_{}_lr_{}_decay_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, 
           opt.batch_size, opt.epochs,
           opt.alpha,opt.beta, opt.gamma, 
           opt.learning_rate, opt.weight_decay, 
           opt.temp, opt.trial )

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.alpha /= (opt.batch_size**2)
    opt.beta /= opt.batch_size
        
    return opt


class PoseDataset(data.Dataset):
    def __init__(self, path1, path2):
        super(PoseDataset,self).__init__()
        self.data = []
        
        # 要构造两个
        self.mytransform = TwoCropTransform(transforms.Compose([
            transforms.Resize(size=(112,112)),
            # transforms.RandomResizedCrop(size=112, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(img_norm_cfg['mean'],img_norm_cfg['std']),
        ]))

        self.to_exp={
            'SU':0, 'AF':1, 'DI':2, 'HA':3, 'SA':4, 'AN':5, 'NE':6
        }
        self.to_pos={
            'FL':0, 'HL':1, 'S':2, 'HR':3, 'FR':4
        }
    
        self.read_data(path1)
        self.read_data(path2)
        

        # print 标签 , 好知道进度
        label1=[x for _,x,_ in self.data]
        label2=[x for _,_,x in self.data]
        print(set(label1))
        print(set(label2))

    def read_data(self,path):
        root_dir = os.path.join(os.getcwd(),path)
        dir_list = os.listdir(root_dir)

        for dir_name in dir_list:
            tmp_dir = os.path.join(os.path.join(root_dir,dir_name))
            img_list = os.listdir(tmp_dir)
            for img_name in img_list:
                t_img = self.mytransform(Image.open(os.path.join(tmp_dir,img_name)).convert("RGB"))
                print('\r{},{},{}'.format(img_name,type(t_img),t_img[0].shape),end='')
                # import pdb; pdb.set_trace()
                id_label1 = self.to_exp[img_name[4:6]]
                id_label2 = int(img_name[2:4])
                id_label2 = id_label2*7+id_label1
                self.data.append((t_img,id_label1,id_label2))
                
        print('\n')
        
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)



def set_model(opt):    
    model = IRes_50()
    
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            # model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    criterion2 = BTLoss(alpha = opt.alpha, batch_size=opt.batch_size)

    criterion3 = nn.CrossEntropyLoss()
    
    return model, criterion, criterion2, criterion3


def train(train_loader, model, criterion, criterion2, criterion3, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    Lcros = AverageMeter()
    Lsups = AverageMeter()
    Lviews = AverageMeter()
    Lbts = AverageMeter()

    end = time.time()


    for idx, (images, label1s, label2s) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            label1s = label1s.cuda(non_blocking=True)
            label2s = label2s.cuda(non_blocking=True)
        bsz = label1s.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        mid1 , mid2 , mid3, res = model(images)
        # print("visit here")
        # print(mid1.shape,mid2.shape,mid3.shape)


        features = nn.functional.normalize(mid3,dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        labels = torch.cat((label1s,label1s))
        Lcro = criterion3(res,labels)
        
        Lsup = criterion(features,label1s)
        Lview = criterion(features,label2s)

        Lbt = criterion2(mid1[:opt.batch_size], mid1[opt.batch_size:])+criterion2(mid2[:opt.batch_size], mid2[opt.batch_size:])
        
        loss = Lcro
        # print("loss={}".format(loss))
        
        # update metric
        Lcros.update(Lcro.item(), bsz)
        Lsups.update(Lsup.item(), bsz)
        Lviews.update(Lview.item(), bsz)
        Lbts.update(Lbt.item(), bsz)
        losses.update(loss.item(), bsz)    
    
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # if (idx + 1) % opt.print_freq == 0:
    print('Train: [{0}][{1}/{2}]\t'
          'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'lcro {lcros.val:.3f} ({lcros.avg:.3f})\t'
          'lsup {lsups.val:.3f} ({lsups.avg:.3f})\t'
          'lview {lviews.val:.3f} ({lviews.avg:.3f})\t'
          'lbt {lbts.val:.3f} ({lbts.avg:.3f})\t'
          'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
           epoch, idx + 1, len(train_loader), batch_time=batch_time,
           lcros=Lcros, lsups=Lsups, lviews=Lviews, lbts=Lbts,
           loss=losses))
        #     sys.stdout.flush()

    return losses.avg

    return losses.avg

def test(train_loader, model, criterion3, epoch):
    """one epoch training"""
    model.eval()

    losses = AverageMeter()
    total =0
    correct = 0

    for idx, (images, label1s, label2s) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            label1s = label1s.cuda(non_blocking=True)
            label2s = label2s.cuda(non_blocking=True)
        bsz = label1s.shape[0]

        # compute loss
        mid1 , mid2 , mid3, res = model(images)

        # cross loss
        labels = torch.cat((label1s,label1s))
        # print(res.shape,labels.shape)
        # print(set(list(labels)))        
        # import pdb; pdb.set_trace()

        Lcro = criterion3(res,labels)       

        # update metric
        losses.update(Lcro.item(), bsz)

        # compute ac.cu
        _,predicted =torch.max(res.data,1)  #return 维度1上的最大值及其索引
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()     

    # print info
    print('Train: [{0}][{1}/{2}]\t'
            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            'accu {accu:.2f}%'.format(
            epoch, idx + 1, len(train_loader), loss=losses,
            accu=correct*100.0/total))
    sys.stdout.flush()

    return 


def main():
    opt = parse_option()

    # build data loader
    with open('datasets/kdef/kdef_norm_train.pkl','rb') as f:
        train_loader = pickle.load(f)
    
    with open('datasets/kdef/kdef_norm_test.pkl','rb') as f:
        test_loader = pickle.load(f)

    # build model and criterion
    model, criterion, criterion2 ,criterion3 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, criterion2, criterion3, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # perform test
        if(epoch%5 ==0):
            test(test_loader, model, criterion3, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'cl_kdef_last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()