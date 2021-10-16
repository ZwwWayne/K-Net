_base_ = [
    '../_base_/models/knet_s3_r50_fpn_panoptic.py',
    '../common/mstrain_3x_coco_panoptic.py'
]

pretrained = 'https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth'
model = dict(
    type='KNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        mlp_ratios=(4, 4, 4, 4),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[64, 128, 320, 512]))

data = dict(samples_per_gpu=1, workers_per_gpu=1)
optimizer = dict(lr=0.0001 / 1.4)

custom_imports = dict(
    imports=[
        'knet.det.knet',
        'knet.det.kernel_head',
        'knet.det.kernel_iter_head',
        'knet.det.kernel_update_head',
        'knet.det.semantic_fpn_wrapper',
        'knet.kernel_updator',
        'knet.det.mask_hungarian_assigner',
        'knet.det.mask_pseudo_sampler',
    ],
    allow_failed_imports=False)
