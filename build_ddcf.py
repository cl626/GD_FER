# 2800*2
from torch.utils import data
from torchvision import transforms, utils, models
from util import TwoCropTransform, AverageMeter
import os
from PIL import Image
import pickle
import torch
from pdb import set_trace
import shutil


dir_list = ['40Females','40Males']

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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

train_size = int(2800*0.85)
valid_size = 2800-train_size

import random
idxset = random.sample(range(2800),train_size)
print(idxset)
    
train_sign = [False]*2800
test_sign = [True]*2800

for idx in idxset:
    train_sign[idx] = True
    test_sign[idx] = False
    
train_set = DDCFDataset('datasets/DDCF/40Males','datasets/DDCF/40Females',train_sign)
test_set = DDCFDataset('datasets/DDCF/40Males','datasets/DDCF/40Females',test_sign)

batch_size = 256

train_loader1 = data.DataLoader(train_set,batch_size=batch_size,drop_last=True,shuffle=True)
test_loader1 = data.DataLoader(test_set,batch_size=batch_size,drop_last=True,shuffle=True)

with open('datasets/ddcf/ddcf_norm_train_256.pkl','wb') as f:
    pickle.dump(train_loader1,f)
    
with open('datasets/ddcf/ddcf_norm_test_256.pkl','wb') as f:
    pickle.dump(test_loader1,f)
    
batch_size = 32

train_loader2 = data.DataLoader(train_set,batch_size=batch_size,drop_last=True)
test_loader2 = data.DataLoader(test_set,batch_size=batch_size,drop_last=True)

with open('datasets/ddcf/ddcf_norm_train_32.pkl','wb') as f:
    pickle.dump(train_loader2,f)
    
with open('datasets/ddcf/ddcf_norm_test_32.pkl','wb') as f:
    pickle.dump(test_loader2,f)