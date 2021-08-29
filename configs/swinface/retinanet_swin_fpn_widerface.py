_base_ = [
    '../_base_/models/retinanet_swin_fpn.py', '../_base_/datasets/wider_face_640x640.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=1))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 20])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=24)
log_config = dict(interval=1)
