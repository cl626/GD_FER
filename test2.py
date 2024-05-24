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

import matplotlib.pyplot as plt

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
                        choices=['ddcf', 'kdef', 'rafdb'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=112, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'FineTune', 'Cross'], help='choose method')

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
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='efficient of loss_BT in loss')
    parser.add_argument('--theta', type=float, default=1.0,
                        help='ratio of loss*3')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='efficient of loss_BT in loss')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='efficient of loss_BT in loss') 
    parser.add_argument('--special', type=str, default='None',
                        help='illustration of special handling')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='ratio of loss*3')    
    
    opt = parser.parse_args()

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
    
class DDCFDataset(data.Dataset):
    def __init__(self, img_dir, label_path):
        super(DDCFDataset,self).__init__()
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

        self.pair = []
        with open(label_path,'r') as file:
            lines =  file.readlines()
            for idx,line in enumerate(lines):
                # if(idx%1000==999):  print(idx)
                img_path, label = line.split()
                img_path = img_path[:img_path.find('.')]+'_aligned'+img_path[img_path.find('.'):]
                self.pair.append([img_path,int(label)])

        self.data =[]
        for i in range(len(self.pair)):
            if(i%1000==999):    print(i)
            self.data.append((self.mytransform(Image.open(os.path.join(img_dir,self.pair[i][0])).convert("RGB")),self.pair[i][1]))

        # print 标签 , 好知道进度
        # label =[x for _,x in self.data]
        # print(set(label))
        
        
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

    
def set_model(opt):    
    if(opt.model=='IR_50'):
        model = IRes_50()
        if(opt.special=='Contra'):
            param_dict =torch.load('save/SupCon/rafdb_IR_50_standard_models/SupCon_rafdb_IR_50_bsz_32_epoch_100_ratio_0.001_alpha_0.0_beta_0.001_gamma_0.5_lr_0.001_decay_0.0005_temp_0.07_trial_0_Contra_cosine/cl_rafdb_last.pth')['model']   # contra
        elif(opt.special=='NoContra'):
            param_dict = torch.load('save/SupCon/rafdb_IR_50_standard_models/Cross_rafdb_IR_50_bsz_32_epoch_100_alpha_0.0_beta_0.001_gamma_0.5_lr_0.001_decay_0.0001_temp_0.07_trial_0_cosine/cl_rafdb_last.pth')['model']   # no contra            

    elif(opt.model=='APViT'):
        model = APViT()        
        if(opt.special=='Contra'):
            param_dict =torch.load('save/SupCon/rafdb_APViT_standard_models/Cross_rafdb_APViT_bsz_32_epoch_100_alpha_0.0_beta_0.001_gamma_0.5_lr_0.001_decay_0.0005_temp_0.07_trial_0_IRes50_cosine/cl_rafdb_last.pth')['model']   # contra
        elif(opt.special=='NoContra'):
            param_dict = torch.load('save/SupCon/rafdb_APViT_standard_models/Cross_rafdb_APViT_bsz_32_epoch_100_alpha_0.0_beta_0.001_gamma_0.5_lr_0.001_decay_0.0005_temp_0.07_trial_0_NoContra_cosine/cl_rafdb_last.pth')['model']   # no contra            
        
    tar_dict ={}
    # param_dict = torch.load('save/SupCon/ddcf_APViT_models/contrast_Cross_ddcf_APViT_bsz_32_epoch_200_lr_5e-05_decay_0.0005_bsz_32_temp_0.1_trial_0_contrast_cosine/ddcf_epoch_50.pth')['model'] # ddcf

    for key,val in param_dict.items():
        if(key[:7]=='module.'):
            tar_dict[key[7:]] = val
        else:
            tar_dict[key]=val
    model.load_state_dict(tar_dict)
    
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


# 用条件逻辑 编程，避免繁琐的手动修改超参
def test(opt,train_loader, model, criterion):
    """one epoch training"""
    model.eval()

    losses = AverageMeter()
    total =0
    correct = 0

    label_true = []
    label_pred = []
    for idx, (images, labels) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        if(opt.model=='IR_50'):
            _, _, _, features = model(images)
        else:
            features = model(images)    # merely for resnet50 & original APViT
        labels = torch.cat([labels,labels])

        loss = criterion(features,labels)
        # update metric
        losses.update(loss.item(), bsz)

        # compute accu
        _,predicted =torch.max(features.data,1)  #return 维度1上的最大值及其索引
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()     

        label_true += labels.cpu().numpy().tolist()
        label_pred += predicted.cpu().numpy().tolist()

    print('accu={:.2f}%'.format(correct*100.0/total))
    return label_true, label_pred

import seaborn as sebrn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as atlas
import random

def draw(opt, y_true, y_pred):    
    print(type(y_true[0]),type(y_pred[0]))
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    conf_matrix = [conf_matrix[i]*1.0/sum(conf_matrix[i]) for i in range(len(conf_matrix))]

    # Using Seaborn heatmap to create the plot
    fx = sebrn.heatmap(conf_matrix, annot=True, cmap="turbo")

    # labels the title and x, y axis of plot
    fx.set_title("{}_{}_{}".format(opt.dataset,opt.model,opt.special))
    fx.set_xlabel("Predicted Values")
    fx.set_ylabel("Actual Values ")

    # labels the boxes
    fx.xaxis.set_ticklabels(['suprised', 'afraid', 'disgust', 'happy', 'sad', 'angry', 'neutral'])
    fx.yaxis.set_ticklabels(['suprised', 'afraid', 'disgust', 'happy', 'sad', 'angry', 'neutral'])

    atlas.show()
    atlas.savefig('pic/conf_matrix/{}_{}_{}.png'.format(opt.dataset,opt.model,opt.special))
    atlas.close()

def main():
    opt = parse_option()

    # build data loader
    # train_loader = set_loader(opt)
    with open('datasets/{dataset}/{dataset}_norm_train.pkl'.format(dataset=opt.dataset),'rb') as f:
        train_loader = pickle.load(f)
    
    with open('datasets/{dataset}/{dataset}_norm_test.pkl'.format(dataset=opt.dataset),'rb') as f:
        test_loader = pickle.load(f)
    
    # build model and criterion
    model, criterion, criterion2, criterion3 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    true, predicted = test(opt,test_loader, model, criterion3)
    
    draw(opt, true, predicted)



if __name__ == '__main__':
    main()
