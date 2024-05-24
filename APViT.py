# build APViT block
from mmcv.runner import load_state_dict
import torch
from torch import nn
from torchsummary import summary
from backbone import IRSE
import torch.nn.functional as F
from vit.PoolingViT import PoolingViT

vit_small = dict(img_size=112, patch_size=16, embed_dim=768, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6)

class IRes_50(nn.Module):
    def __init__(self, num_classes=7, pret='weights/backbone_ir50_ms1m_epoch120.pth',model='IR_50'):
        super(IRes_50, self).__init__()
        
        self.model = model
        print(self.model)
        
        self.net = IRSE(
                input_size=(112, 112),
                num_layers=50,
                pretrained=pret,
                mode='ir',
                with_head=True,
                return_index=[1,2,3],   # only use the first 3 stages
                return_type='Tuple',
        )

        # self.net.output_layer.add_module('Acti',nn.ReLU())
        # self.net.output_layer.add_module('Linear',nn.Linear(512,num_classes))    
        # self.net.output_layer.add_module('Bn',nn.BatchNorm1d(num_classes))
        self.net.output_layer[3] = nn.Linear(512*7*7,7)
        self.net.output_layer[4] = nn.BatchNorm1d(7)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        )

        self.head2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        )

        self.head3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        )

        #(-1,128,28,28)->>(-1,512,7,7)
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(512)
        )
        
        #(-1,256,14,14)->>(-1,512,7,7)
        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(512)
        )
        
        
    def forward(self, x):
        mid1, mid2, mid3, out = self.net(x)
        inter1 = self.head2(self.proj2(mid1))
        inter2 = self.head3(self.proj3(mid2))
        inter3 = self.head(mid3)
        
        if(self.model == 'IR_50'):
            return inter1, inter2, inter3, out
        else:
            return mid1, mid2, mid3, out   # for apvit
    
class APViT(nn.Module):
    def __init__(self, num_classes=7,pret=None,model='APViT'):
        super(APViT, self).__init__()

        if(pret!=None):
            self.backbone = IRes_50(pret=None,model=model)
            print('use contrast?')  
            self.backbone.load_state_dict({k.replace('module.',''):v for k,v in torch.load(pret)['model'].items()})   # for raf-db
        else:
            self.backbone = IRes_50(model=model)
            
        self.apvit = PoolingViT(
                pretrained='weights/vit_small_p16_224-15ec54c9.pth',
                input_type='feature',
                patch_num=196,
                in_channels=[256],
                attn_method='SUM_ABS_1',
                sum_batch_mean=False,
                cnn_pool_config=dict(keep_num=160, exclude_first=False),
                vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 4, exclude_first=True, attn_method='SUM'),  # None by default
                depth=8,
                **vit_small,
        )

        self.clshead = nn.Linear(768,num_classes,bias=True)

    def forward(self, x):
        mid1, mid2, mid3, out = self.backbone(x)
        res = self.apvit(tuple([mid2]))
        return self.clshead(res['x'])
        # out = F.normalize(self.clshead(res['x']))
        # return out