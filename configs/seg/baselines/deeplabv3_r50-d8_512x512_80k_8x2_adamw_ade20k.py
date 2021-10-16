_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_adamw_80k.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))
data = dict(samples_per_gpu=2, workers_per_gpu=2)
