_base_ = [
    '../_base_/models/knet_s3_r50_fpn_panoptic.py',
    '../common/mstrain_3x_coco_panoptic.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
model = dict(
    type='KNet',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]))

# update the parameter wise config according to Swin for mask rcnn
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

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
