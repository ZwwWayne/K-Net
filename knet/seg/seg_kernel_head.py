import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads import ASPPHead, FCNHead, PSPHead, UPerHead
from mmseg.ops import resize


@HEADS.register_module()
class ASPPKernelHead(ASPPHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, **kwargs):
        super(ASPPKernelHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        feats = self.bottleneck(aspp_outs)
        output = self.cls_seg(feats)
        seg_kernels = self.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())
        return output, feats, seg_kernels


@HEADS.register_module()
class UPerKernelHead(UPerHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, **kwargs):
        super(UPerKernelHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(feats)

        seg_kernels = self.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())
        return output, feats, seg_kernels


@HEADS.register_module()
class PSPKernelHead(PSPHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, **kwargs):
        super(PSPKernelHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        feats = self.bottleneck(psp_outs)
        output = self.cls_seg(feats)

        seg_kernels = self.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())
        return output, feats, seg_kernels


@HEADS.register_module()
class FCNKernelHead(FCNHead):

    def __init__(self, **kwargs):
        super(FCNKernelHead, self).__init__(**kwargs)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        output = self.cls_seg(feats)

        seg_kernels = self.conv_seg.weight.clone()
        seg_kernels = seg_kernels[None].expand(
            feats.size(0), *seg_kernels.size())
        return output, feats, seg_kernels
