_base_ = [
    '../_base_/models/knet_s3_r50_fpn_panoptic.py',
    '../common/mstrain_3x_coco_panoptic.py'
]

model = dict(
    backbone=dict(
        depth=101,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))
