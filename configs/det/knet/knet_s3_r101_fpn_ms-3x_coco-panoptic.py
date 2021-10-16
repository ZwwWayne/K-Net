_base_ = [
    '../_base_/models/knet_s3_r50_fpn_panoptic.py',
    '../common/mstrain_3x_coco_panoptic.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))
