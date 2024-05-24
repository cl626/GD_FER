from torch.utils import data
from torchvision import transforms, utils, models
from util import TwoCropTransform, AverageMeter
import os
from PIL import Image
import pickle
import torch
from pdb import set_trace
import shutil


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
                # print(label)
                img_path = img_path[:img_path.find('.')]+'_aligned'+img_path[img_path.find('.'):]
                self.pair.append([img_path,int(label)-1])

        self.data =[]
        for i in range(len(self.pair)):
            if(i%1000==999):    print(i)
            self.data.append((self.mytransform(Image.open(os.path.join(img_dir,self.pair[i][0])).convert("RGB")),self.pair[i][1]))

        # print 标签 , 好知道进度
        label =[x for _,x in self.data]
        print(set(label))
        
        
    def __getitem__(self,index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

train_set = DDCFDataset('datasets/rafdb/aligned','datasets/rafdb/EmoLabel/train.txt')
test_set = DDCFDataset('datasets/rafdb/aligned','datasets/rafdb/EmoLabel/test.txt')
print(len(train_set))

# import pdb; pdb.set_trace()

batch_size = 256

train_loader = data.DataLoader(train_set,batch_size=batch_size,drop_last=True,shuffle=True,pin_memory=False)
test_loader = data.DataLoader(test_set,batch_size=batch_size,drop_last=True,shuffle=True,pin_memory=False)

with open('datasets/rafdb/rafdb_norm_train_{}.pkl'.format(batch_size),'wb') as f:
    pickle.dump(train_loader,f)
    
with open('datasets/rafdb/rafdb_norm_test_{}.pkl'.format(batch_size),'wb') as f:
    pickle.dump(test_loader,f)