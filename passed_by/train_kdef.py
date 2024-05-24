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
                        choices=['cifar10', 'cifar100', 'path', 'kdef'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=112, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='Cross',
                        choices=['SupCon', 'SimCLR', 'Cross'], help='choose method')

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
    parser.add_argument('--resnet50', type=str, default='1',
                        help='using cosine annealing')
    
    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'label2_{}3_{}_{}_resnet50_{}_bsz_{}_epoch_{}alpha_{}_beta_{}_gamma_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, 
               opt.resnet50, opt.batch_size, opt.epochs,
               opt.alpha,opt.beta, opt.gamma, 
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


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


class PoseDataset(data.Dataset):
    def __init__(self, path):
        super(PoseDataset,self).__init__()
        root_dir = os.path.join(os.getcwd(),path)
        dir_list = os.listdir(root_dir)
        self.data = []
        
        # 要构造两个
        self.mytransform = TwoCropTransform(transforms.Compose([
            transforms.Resize(size=112),
            # transforms.RandomResizedCrop(size=112, scale=(0.2, 1.)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
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
    with open('kdef_simple.pkl','rb') as f:
        mydataset=pickle.load(f)
    # mydataset = PoseDataset('datasets/KDEF/AF')
    # with open('kdef_plain.pkl','wb') as f:
    #     pickle.dump(mydataset,f)
    # import pdb;  pdb.set_trace()
    # print(type(mydataset[0][0]))
    train_size = int(len(mydataset)*0.8)
    valid_size = len(mydataset)-train_size
    # print(train_size,valid_size)
    trainset,validset=torch.utils.data.random_split(mydataset,[train_size,valid_size])

    myloader = data.DataLoader(mydataset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
    train_loader = data.DataLoader(trainset,batch_size=opt.batch_size,drop_last=True,shuffle=True)
    test_loader = data.DataLoader(validset,batch_size=opt.batch_size,drop_last=True,shuffle=True)

    return train_loader,test_loader

def set_model(opt):
    # 用CL的预训练weight
    model = SupConResNet2(name=opt.model)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./save/SupCon/kdef_models/label2__SupCon_kdef_resnet50_resnet50_1_alpha_0.5_bigbeta_0.03_gamma_0.5_bsz_256_temp_0.1_trial_0_cosine/cl_kdef_last.pth')['model'].items()})
    model.head=nn.Sequential(
        nn.Linear(2048,128),
        nn.ReLU(inplace=True),
        nn.Linear(128,7),
    )
    # model.head.add_module("linear1",nn.Linear(128,32))
    # model.head.add_module("relu",nn.ReLU())
    # model.head.add_module("linear2",nn.Linear(32,7))
    
    # resnet50自带的预训练weight
    # model = models.resnet50(pretrained=True) #pretrained=True 加载模型以及训练过的参数
    # print(model.fc)
    # model.fc = nn.Sequential(
    #     nn.Linear(2048,128),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(128,7),
    # )
    
    
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
        _, _, features = model(images)
        # features = model(images)      # merely for resnet50
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
        _, _, features = model(images)
        # features = model(images)    # merely for resnet50
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

    # build data loader
    # train_loader = set_loader(opt)
    train_loader,test_loader = set_loader_kdef(opt)
    
    # build model and criterion
    model, criterion, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


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
