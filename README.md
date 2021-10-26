# K-Net: Towards Unified Image Segmentation

## Introduction

This is an official release of the paper **K-Net:Towards Unified Image Segmentation**. K-Net will also be integrated in the future release of MMDetection and MMSegmentation.

> [**K-Net:Towards Unified Image Segmentation**](https://arxiv.org/abs/2106.14855),
> Wenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy
> In: Proc. Advances in Neural Information Processing Systems (NeurIPS), 2021
> *arXiv preprint ([arXiv 2106.14855](https://arxiv.org/abs/2106.14855))*

## Results

The model checkpoints and logs will be released soon.

### Semantic Segmentation on ADE20K

| Backbone | Method | Crop Size | Lr Schd | mIoU | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| R-50 | K-Net + FCN | 512x512 | 80K | 43.3 |[config](configs/seg/knet/knet_s3_fcn_r50-d8_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| R-50 | K-Net + PSPNet | 512x512 | 80K | 43.9 |[config](configs/seg/knet/knet_s3_pspnet_r50-d8_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| R-50 | K-Net + DeepLabv3 | 512x512 | 80K | 44.6 |[config](configs/seg/knet/knet_s3_deeplabv3_r50-d8_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| R-50 | K-Net + UPerNet | 512x512 | 80K | 43.6 |[config](configs/seg/knet/knet_s3_upernet_r50-d8_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| Swin-T | K-Net + UPerNet | 512x512 | 80K | 45.4 |[config](configs/seg/knet/knet_s3_upernet_swin-t_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| Swin-L | K-Net + UPerNet | 512x512 | 80K | 52.0 |[config](configs/seg/knet/knet_s3_upernet_swin-l_80k_adamw_ade20k.py) | [model]() &#124;  [log]() |
| Swin-L | K-Net + UPerNet | 640x640 | 80K | 52.7 |[config](configs/seg/knet/knet_s3_upernet_swin-l_80k_adamw_640x640_ade20k.py) | [model]() &#124;  [log]() |

### Instance Segmentation on COCO

| Backbone | Method | Lr Schd | Mask mAP| Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50  | K-Net | 1x        | 34.0 |[config](configs/det/knet/knet_s3_r50_fpn_1x_coco.py) | [model]() &#124;  [log]() |
| R-50  | K-Net | ms-3x     | 37.8 |[config](configs/det/knet/knet_s3_r50_fpn_ms-3x_coco.py) | [model]() &#124;  [log]() |
| R-101  | K-Net | ms-3x    | 39.2 |[config](configs/det/knet/knet_s3_r101_fpn_ms-3x_coco.py) | [model]() &#124;  [log]() |
| R-101-DCN | K-Net | ms-3x | 40.5 |[config](configs/det/knet/knet_s3_r101_dcn-c3-c5_fpn_ms-3x_coco.py) | [model]() &#124;  [log]() |

### Panoptic Segmentattion on COCO

| Backbone | Method | Lr Schd | PQ | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50  | K-Net | 1x| 44.3 |[config](configs/det/knet/knet_s3_r50_fpn_1x_coco-panoptic.py) | [model]() &#124;  [log]() |
| R-50  | K-Net | ms-3x| 47.1 |[config](configs/det/knet/knet_s3_r50_fpn_ms-3x_coco-panoptic.py) | [model]() &#124;  [log]() |
| R-101  | K-Net | ms-3x| 48.4 |[config](configs/det/knet/knet_s3_r101_fpn_ms-3x_coco-panoptic.py) | [model]() &#124;  [log]() |
| R-101-DCN  | K-Net | ms-3x| 49.6 |[config](configs/det/knet/knet_s3_r101_dcn-c3-c5_fpn_ms-3x_coco-panoptic.py) | [model]() &#124;  [log]() |
| Swin-L (window size 7)  | K-Net | ms-3x| 54.6 |[config](configs/det/knet/knet_s3_swin-l_fpn_ms-3x_16x2_coco-panoptic.py) | [model]() &#124;  [log]() |
| Above on test-dev  | | | 55.2 | | |

## Installation

It requires the following OpenMMLab packages:

- MIM >= 0.1.5
- MMCV-full >= v1.3.14
- MMDetection >= v2.17.0
- MMSegmentation >= v0.18.0
- scipy
- panopticapi

```bash
pip install openmim scipy mmdet mmsegmentation
pip install git+https://github.com/cocodataset/panopticapi.git
mim install mmcv-full
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Usage

### Data preparation

Prepare data following [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). The data structure looks like below:

```
data/
├── ade
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   ├── images
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   │   ├── panoptic_{train,val}2017/  # panoptic png annotations
│   │   ├── image_info_test-dev2017.json  # for test-dev submissions
│   ├── train2017
│   ├── val2017
│   ├── test2017

```

### Training and testing

For training and testing, you can directly use mim to train and test the model

```bash
# train instance/panoptic segmentation models
sh ./tools/mim_slurm_train.sh $PARTITION mmdet $CONFIG $WORK_DIR

# test instance segmentation models
sh ./tools/mim_slurm_test.sh $PARTITION mmdet $CONFIG $CHECKPOINT --eval segm

# test panoptic segmentation models
sh ./tools/mim_slurm_test.sh $PARTITION mmdet $CONFIG $CHECKPOINT --eval pq

# train semantic segmentation models
sh ./tools/mim_slurm_train.sh $PARTITION mmseg $CONFIG $WORK_DIR

# test semantic segmentation models
sh ./tools/mim_slurm_test.sh $PARTITION mmseg $CONFIG $CHECKPOINT --eval mIoU
```

For test submission for panoptic segmentation, you can use the command below:

```bash
# we should update the category information in the original image test-dev pkl file
# for panoptic segmentation
python -u tools/gen_panoptic_test_info.py
# run test-dev submission
sh ./tools/mim_slurm_test.sh $PARTITION mmdet $CONFIG $CHECKPOINT  --format-only --cfg-options data.test.ann_file=data/coco/annotations/panoptic_image_info_test-dev2017.json data.test.img_prefix=data/coco/test2017 --eval-options jsonfile_prefix=$WORK_DIR
```

You can also run training and testing without slurm by directly using mim for instance/semantic/panoptic segmentation like below:

```bash
PYTHONPATH='.':$PYTHONPATH mim train mmdet $CONFIG $WORK_DIR
PYTHONPATH='.':$PYTHONPATH mim train mmseg $CONFIG $WORK_DIR
```

- PARTITION: the slurm partition you are using
- CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CONFIG: the config files under the directory `configs/`
- JOB_NAME: the name of the job that are necessary for slurm

## Citation

```
@inproceedings{zhang2021knet,
    title={{K-Net: Towards} Unified Image Segmentation},
    author={Wenwei Zhang and Jiangmiao Pang and Kai Chen and Chen Change Loy},
    year={2021},
    booktitle={NeurIPS},
}
```
