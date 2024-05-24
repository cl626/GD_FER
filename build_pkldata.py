from torch.utils import data
from torchvision import transforms, utils, models
from util import TwoCropTransform, AverageMeter
import os
from PIL import Image
import pickle
import torch

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)




class PoseDataset(data.Dataset):
    def __init__(self, path1, path2,sign_list):
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
    
        self.read_data(path1,sign_list)
        self.read_data(path2,sign_list)
        

        # print 标签 , 好知道进度
        label1=[x for _,x,_ in self.data]
        label2=[x for _,_,x in self.data]
        print(set(label1))
        print(set(label2))

    def read_data(self,path,sign_list):
        root_dir = os.path.join(os.getcwd(),path)
        dir_list = os.listdir(root_dir)

        for idx1,dir_name in enumerate(dir_list):
            tmp_dir = os.path.join(os.path.join(root_dir,dir_name))
            img_list = os.listdir(tmp_dir)
            for idx2,img_name in enumerate(img_list):
                if(sign_list[idx1*35+idx2]==False):
                    continue
                t_img = self.mytransform(Image.open(os.path.join(tmp_dir,img_name)).convert("RGB"))
                print('\r{},{},{}'.format(img_name,type(t_img),t_img[0].shape),end='')
                # import pdb; pdb.set_trace()
                id_label1 = self.to_exp[img_name[4:6]]
                id_label2 = int(img_name[2:4])
                id_label2 = id_label2*7+id_label1-6
                self.data.append((t_img,id_label1,id_label2))
                
        print('\n')
        
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    
train_size = int(2450*0.85)
valid_size = 2450-train_size

import random
idxset = random.sample(range(2450),train_size)
print(idxset)
    
train_sign = [False]*2450
test_sign = [True]*2450

for idx in idxset:
    train_sign[idx] = True
    test_sign[idx] = False
    
import numpy as np
train_set = PoseDataset('datasets/KDEF/AF','datasets/KDEF/BF',train_sign)
test_set = PoseDataset('datasets/KDEF/AF','datasets/KDEF/BF',test_sign)
# import pdb;  pdb.set_trace()

batch_size = 512
                       
# trainset,validset=torch.utils.data.random_split(mydataset,[train_size,valid_size])
# train_set = torch.tensor(mydataset[train_sign])
# test_set = torch.tensor(mydataset[test_sign])

train_loader1 = data.DataLoader(train_set,batch_size=batch_size,drop_last=True)
test_loader1 = data.DataLoader(test_set,batch_size=batch_size,drop_last=True)


# with open('datasets/kdef/kdef_plain.pkl','wb') as f:
#     pickle.dump(mydataset,f)
    
# with open('datasets/kdef/kdef_norm.pkl','wb') as f:
#     pickle.dump(mydataset,f)

# with open('datasets/kdef/kdef_extreme.pkl','wb') as f:
#     pickle.dump(mydataset,f)

with open('datasets/kdef/kdef_norm_train_{}.pkl'.format(batch_size),'wb') as f:
    pickle.dump(train_loader1,f)
    
with open('datasets/kdef/kdef_norm_test_{}.pkl'.format(batch_size),'wb') as f:
    pickle.dump(test_loader1,f)
    

# batch_size = 32
# train_loader2 = data.DataLoader(train_set,batch_size=batch_size,drop_last=True)
# test_loader2 = data.DataLoader(test_set,batch_size=batch_size,drop_last=True)

# with open('datasets/kdef/kdef_norm_train_32.pkl','wb') as f:
#     pickle.dump(train_loader2,f)
    
# with open('datasets/kdef/kdef_norm_test_32.pkl','wb') as f:
#     pickle.dump(test_loader2,f)