# import torch
# from networks.resnet_big import SupConResNet

# pkg = torch.load('./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_64_temp_0.1_trial_0_cosine/last.pth')
# with open('origin_param.txt','w') as file:
#     file.write(str(pkg['model'].keys()))


#model=torch.load('./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_64_temp_0.1_trial_0_cosine/last.pth')['model']
    

from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch import nn

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big2 import SupConResNet, SupConResNet2
from losses import SupConLoss, BTLoss
from APViT import APViT,IRes_50

# for kdef
from torch.utils import data
from torchvision import transforms, utils, models
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='IR_50')
    parser.add_argument('--dataset', type=str, default='kdef',
                        choices=['ddcf', 'rafdb', 'kdef'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=112, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='Cross',
                        choices=['SupCon', 'Combine', 'Cross'], help='choose method')

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
    parser.add_argument('--beta', type=float, default='0.5',
                        help='efficient of loss_BT in loss')
    parser.add_argument('--gamma', type=float, default='0.5',
                        help='efficient of loss_BT in loss') 
    parser.add_argument('--special', type=str, default='None',
                        help='illustration of special handling')
    
    
    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_{}_standard_models'.format(opt.dataset,opt.model)   #保存到指定路径
    opt.tb_path = './save/SupCon/{}_{}_standard_tensorboard'.format(opt.dataset,opt.model)  #保存tb，一直用不起来

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if(opt.method == 'Cross'):
        opt.model_name = '{}_{}_{}_{}_bsz_{}_epoch_{}_lr_{}_decay_{}_temp_{}_trial_{}'.\
        format(opt.special, opt.method, opt.dataset, opt.model, 
               opt.batch_size, opt.epochs,
               opt.learning_rate, opt.weight_decay,
               opt.temp, opt.trial )
    elif(opt.method == 'Combine'):
        opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_alpha_{}_beta_{}_gamma_{}_lr_{}_decay_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, 
               opt.batch_size, opt.epochs,
               opt.alpha,opt.beta, opt.gamma, 
               opt.learning_rate, opt.weight_decay,
               opt.temp, opt.trial )
    else:
        opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_alpha_{}_beta_{}_gamma_{}_lr_{}_decay_{}_temp_{}_trial_{}'.\
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

        
    return opt


class PoseDataset(data.Dataset):
    def __init__(self, path):
        super(PoseDataset,self).__init__()
        root_dir = os.path.join(os.getcwd(),path)
        dir_list = os.listdir(root_dir)
        self.data = []
        
        mean=[123.675, 116.28, 103.53]
        std=[58.395, 57.12, 57.375]
        
        # 要构造两个
        self.mytransform = TwoCropTransform(transforms.Compose([
            transforms.Resize(size=(112,112)),
            # transforms.RandomResizedCrop(size=112, scale=(0.2, 1.)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ]))

        self.to_exp={
            'SU':0, 'AF':1, 'DI':2, 'HA':3, 'SA':4, 'AN':5, 'NE':6
        }
        self.to_pos={
            'FL':0, 'HL':1, 'S':2, 'HR':3, 'FR':4
        }
        
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
        # print 标签 , 好知道进度
        label1=[x for _,x,_ in self.data]
        label2=[x for _,_,x in self.data]
        print(set(label1))
        print(set(label2))

    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def set_loader_kdef(opt):

    with open('datasets/kdef/kdef_plain.pkl','rb') as f:
        mydataset=pickle.load(f)
        
    train_size = int(len(mydataset)*0.85)
    valid_size = len(mydataset)-train_size
    # print(train_size,valid_size)
    trainset,validset=torch.utils.data.random_split(mydataset,[train_size,valid_size])

    myloader = data.DataLoader(mydataset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
    train_loader = data.DataLoader(trainset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
    test_loader = data.DataLoader(validset,batch_size=opt.batch_size,drop_last=True,shuffle=True)

    return train_loader,test_loader


def set_model(opt):    
    model = IRes_50()
    # model.load_state_dict(torch.load('save/SupCon/kdef_SupCon_models/SupCon_kdef_IR_50_bsz_32_epoch_200_alpha_0.5_beta_0.02_gamma_0.5_lr_0.05_decay_0.0001_bsz_32_temp_0.1_trial_0_cosine/cl_kdef_last.pth')['model'])
    
    # criterion = SupConLoss(temperature=opt.temp)
    criterion = nn.CrossEntropyLoss()

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

    return model, criterion, criterion2


def train(train_loader, model, criterion, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels, _) in enumerate(train_loader):
        # print(labels.shape)
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        # _, _, features = model(images)
        _, _, _,features = model(images)      # merely for resnet50 & original APViT
        # print(f"shape of features={features.shape}")
        
        labels = torch.cat((labels,labels))
        # print('labels\' type={}, shape={}'.format(type(labels),labels.shape))
        loss = criterion(features,labels)

        # update metric
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
          'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
           epoch, idx + 1, len(train_loader), batch_time=batch_time,
           data_time=data_time, loss=losses))
        #     sys.stdout.flush()

    return losses.avg

def test(train_loader, model, criterion, criterion2, epoch):
    """one epoch training"""
    model.eval()

    losses = AverageMeter()
    total =0
    correct = 0

    for idx, (images, labels, _) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        # _, _, features = model(images)
        _, _, _, features = model(images)    # merely for resnet50 & original APViT
        labels = torch.cat([labels,labels])

        loss = criterion(features,labels)

        # update metric
        losses.update(loss.item(), bsz)

        # compute accu
        _,predicted =torch.max(features.data,1)  #return 维度1上的最大值及其索引
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
    print(opt)

    # build data loader
    # train_loader = set_loader(opt)
    with open('datasets/kdef/kdef_norm_train_256.pkl','rb') as f:
        train_loader = pickle.load(f)
    
    with open('datasets/kdef/kdef_norm_test_256.pkl','rb') as f:
        test_loader = pickle.load(f)
    
    # build model and criterion
    model, criterion, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)    
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # perform test
        if(epoch%5 ==0):
            test(test_loader, model, criterion, criterion2, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
