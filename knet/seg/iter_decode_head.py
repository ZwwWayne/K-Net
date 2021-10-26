import torch.nn as nn
from mmseg.models.builder import HEADS, build_head
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class IterativeDecodeHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self,
                 num_stages,
                 kernel_generate_head,
                 kernel_update_head,
                 in_index=-1,
                 **kwargs):
        super(BaseDecodeHead, self).__init__(**kwargs)
        assert num_stages == len(kernel_update_head)
        self.num_stages = num_stages
        self.kernel_generate_head = build_head(kernel_generate_head)
        self.kernel_update_head = nn.ModuleList()
        self.align_corners = self.kernel_generate_head.align_corners
        self.num_classes = self.kernel_generate_head.num_classes
        self.input_transform = self.kernel_generate_head.input_transform
        self.ignore_index = self.kernel_generate_head.ignore_index
        self.in_index = in_index

        for head_cfg in kernel_update_head:
            self.kernel_update_head.append(build_head(head_cfg))

    def forward(self, inputs):
        """Forward function."""
        sem_seg, feats, seg_kernels = self.kernel_generate_head(inputs)
        stage_segs = [sem_seg]
        for i in range(self.num_stages):
            sem_seg, seg_kernels = self.kernel_update_head[i](feats,
                                                              seg_kernels,
                                                              sem_seg)
            stage_segs.append(sem_seg)
        if self.training:
            return stage_segs
        # only return the prediction of the last stage during testing
        return stage_segs[-1]

    def losses(self, seg_logit, seg_label):
        losses = dict()
        for i, logit in enumerate(seg_logit):
            loss = self.kernel_generate_head.losses(logit, seg_label)
            for k, v in loss.items():
                losses[f'{k}.s{i}'] = v

        return losses
