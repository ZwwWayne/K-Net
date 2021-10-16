_base_ = 'knet_s3_upernet_r50-d8_80k_adamw_ade20k.py'

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_stages = 3
conv_kernel_size = 1

model = dict(
    type='EncoderDecoder',
    pretrained='./pretrain/swin/swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    decode_head=dict(
        kernel_generate_head=dict(
            in_channels=[96, 192, 384, 768])),
    auxiliary_head=dict(
        in_channels=384))

# modify learning rate following the official implementation of Swin Transformer
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
