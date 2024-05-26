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
    parser.add_argument('--model', type=str, default='resnet50')
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
    opt.model_path = './save/SupCon/{}_{}_models'.format(opt.dataset,opt.model)
    opt.tb_path = './save/SupCon/{}_{}_tensorboard'.format(opt.dataset,opt.model)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if(opt.method == 'Cross'):
        opt.model_name = '{}_{}_{}_{}_bsz_{}_epoch_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}_{}'.\
        format(opt.special, opt.method, opt.dataset, opt.model, 
               opt.batch_size, opt.epochs,
               opt.learning_rate, opt.weight_decay, opt.batch_size,
               opt.temp, opt.trial, opt.special )
    elif(opt.method == 'Combine'):
        opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_alpha_{}_beta_{}_gamma_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, 
               opt.batch_size, opt.epochs,
               opt.alpha,opt.beta, opt.gamma, 
               opt.learning_rate, opt.weight_decay, opt.batch_size,
               opt.temp, opt.trial )
    else:
        opt.model_name = '{}_{}_{}_bsz_{}_epoch_{}_alpha_{}_beta_{}_gamma_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
format(opt.method, opt.dataset, opt.model, 
       opt.batch_size, opt.epochs,
       opt.alpha,opt.beta, opt.gamma, 
       opt.learning_rate, opt.weight_decay, opt.batch_size,
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


class DDCFDataset(data.Dataset):
    def __init__(self, path1, path2, sign_list):
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

        self.to_exp={
            'Surprised':0,  'Afraid':1, 'Disgusted':2,  'Happy':3,  'Sad':4,    'Angry':5,  'Neutral':6
        }
        self.to_pos={
            'FL':0, 'HL':1, 'S':2, 'HR':3, 'FR':4
        }
    
        self.read_data(path1, sign_list)
        self.read_data(path2, sign_list)
        

        # print 标签 , 好知道进度
        label1=[x for _,x,_ in self.data]
        label2=[x for _,_,x in self.data]
        print(set(label1))
        print(set(label2))

    def read_data(self,path,sign_list):
        root_dir = os.path.join(os.getcwd(),path)
        # print(root_dir)
        # set_trace()     # Afraid
        dir_list = os.listdir(root_dir)

        for idx1, exp_name in enumerate(dir_list):
            exp_dir = os.path.join(os.path.join(root_dir,exp_name))
            child_list = os.listdir(exp_dir)
            exp_label = self.to_exp[exp_name]
            for idx2, child_name in enumerate(child_list):
                if(child_name == '.DS_Store'):
                    os.remove(os.path.join(exp_dir,child_name))
                    continue
                img_dir = os.path.join(os.path.join(exp_dir,child_name))
                img_list = os.listdir(img_dir)

                for idx3, img_name in enumerate(img_list):
                    if(os.path.isdir(img_name)):
                        shutil.rmtree(os.path.join(img_dir,img_name))
                        continue

                    if(img_name == '.DS_Store' ):
                        os.remove(os.path.join(img_dir,img_name))
                        continue

                    if(sign_list[idx1*400+idx2*10+idx3]==True):
                        t_img = self.mytransform(Image.open(os.path.join(img_dir,img_name)).convert("RGB"))
                        print('\r{},{},{}'.format(img_name,type(t_img),t_img[0].shape),end='')
                        # import pdb; pdb.set_trace()
                        id_label = int(img_name[1:img_name.find('_')])
                        id_label = id_label*7+exp_label-6
                        self.data.append((t_img,exp_label,id_label))
                    else:
                        continue
        print('\n')
        
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
    # model = IRes_50()
    model = APViT(pret='save/SupCon/{dataset}_IR_50_standard_models/SupCon_{dataset}_IR_50_bsz_256_epoch_500_ratio_0.01_alpha_0.0_beta_0.001_gamma_0.5_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/cl_{dataset}_last.pth'.format(dataset=opt.dataset))  # 加预训练
    # model = APViT(pret='save/SupCon/{dataset}_IR_50_standard_models/SupCon_{dataset}_IR_50_bsz_256_epoch_500_ratio_0.0_alpha_0.0_beta_0.001_gamma_0.5_lr_0.01_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine/cl_{dataset}_last.pth'.format(dataset=opt.dataset))  # 不加预训练
    # model = APViT(pret='weights/cl_kdef_last.pth')
    # model = APViT()
    
    
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
        # _, _, _,features = model(images)      # merely for resnet50 & original APViT
        features = model(images)      # merely for resnet50 & original APViT
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
          'loss {loss.val:.6f} ({loss.avg:.6f})'.format(
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
    
    label_true = []
    label_pred = []
    
    for idx, (images, labels, _) in enumerate(train_loader):

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        # _, _, _, features = model(images)    # merely for resnet50 & original APViT
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
        
    # print info
    print('Test: [{0}][{1}/{2}]\t'
            'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            'accu {accu:.2f}%'.format(
            epoch, idx + 1, len(train_loader), loss=losses,
            accu=correct*100.0/total))
    sys.stdout.flush()
    
    return losses.avg, correct*100.0/total, label_true, label_pred

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
    fx.set_title("{}_{}_{}\n\n".format(opt.dataset,opt.model,opt.special))
    fx.set_xlabel("Predicted Values")
    fx.set_ylabel("Actual Values ")

    # labels the boxes
    fx.xaxis.set_ticklabels(['suprised', 'afraid', 'disgust', 'happy', 'sad', 'angry', 'neutral'])
    fx.yaxis.set_ticklabels(['suprised', 'afraid', 'disgust', 'happy', 'sad', 'angry', 'neutral'])

    atlas.show()
    atlas.savefig('pic/conf_matrix/{}_{}.png'.format(opt.dataset,opt.model))

def main():
    opt = parse_option()
    print(opt)

    # build data loader
    with open('datasets/{}/{}_norm_train_32.pkl'.format(opt.dataset,opt.dataset),'rb') as f:
        train_loader = pickle.load(f)
    
    # train_loader.pin_memory = True
    
    with open('datasets/{}/{}_norm_test_32.pkl'.format(opt.dataset,opt.dataset),'rb') as f:
        test_loader = pickle.load(f)
    
    # build model and criterion
    model, criterion, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    
    test_accus = []
    test_losses = []
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        
        # losses = []
        # accus = []

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
            loss , accu, true, pred = test(test_loader, model, criterion, criterion2, epoch)
            test_losses.append(loss)
            test_accus.append(accu)
            if(epoch == opt.epochs):
                draw(opt, true, predicted)                

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, '{dataset}_epoch_{epoch}.pth'.format(dataset=opt.dataset, epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    plt.plot(range(5,opt.epochs+1,5),test_accus)
    plt.xlabel('epoch')
    plt.ylabel('test accu')
    plt.xticks(range(0,opt.epochs+1,5))
    plt.yticks(range(0,101,5))
    plt.grid()    
    plt.savefig(os.path.join('pic/{}/'.format(opt.dataset),'apvit_entire_lr_{}_wd_{}_{}.png'.format(opt.learning_rate,opt.weight_decay,opt.special)))

    accu_data={'epoch':list(range(5,opt.epochs+1,5)),'accu':test_accus, 'loss':test_losses}
    with open(os.path.join('pic/{}'.format(opt.dataset),'apvit_entire_lr_{}_wd_{}_{}.pkl'.format(opt.learning_rate,opt.weight_decay,opt.special)),'wb') as f:
              pickle.dump(accu_data,f)
            
    # save the last model
#     save_file = os.path.join(
#         opt.save_folder, 'last.pth')
#     save_model(model, optimizer, opt, opt.epochs, save_file)
    


if __name__ == '__main__':
    main()
    