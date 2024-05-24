from mmcv.runner import load_state_dict
from .APTP import PoolingViT

vit = PoolingViT(
    pretrained='weights/vit_small_p16_224-15ec54c9.pth',
    input_type='feature',
    patch_num=196,
    in_channels=[256],
    attn_method='SUM_ABS_1',
    sum_batch_mean=False,
    cnn_pool_config=dict(keep_num=160, exclude_first=False),
    vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 4, exclude_first=True, attn_method='SUM'),  # None by default
    depth=8,
)