import torch


def sem2ins_masks(gt_sem_seg,
                  num_thing_classes=80):
    """Convert semantic segmentation mask to binary masks

    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.

    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    # gt_sem_seg is zero-started, where zero indicates the first class
    # since mmdet>=2.17.0, see more discussion in
    # https://mmdetection.readthedocs.io/en/latest/conventions.html#coco-panoptic-dataset  # noqa
    classes = torch.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i < num_thing_classes:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = torch.stack(labels)
        masks = torch.cat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.long(), masks.float()
