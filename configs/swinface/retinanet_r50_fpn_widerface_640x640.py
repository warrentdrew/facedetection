_base_ = [
    #'../_base_/models/retinanet_r50_fpn.py',
    #'../_base_/datasets/wider_face_640x640.py',
    '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = '/raid/datazyp/DATASET/WIDER_voc2/' #'data/WIDERFace/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomSquareCrop',
         crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
    dict(
       type='PhotoMetricDistortionRe',
       brightness_delta=32,
       contrast_range=(0.5, 1.5),
       saturation_range=(0.5, 1.5),
       hue_delta=18),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1100, 1650),
        #scale_factor = [0.3, 0.45, 0.6, 0.8, 1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32, pad_val=0),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=2,
#     train=dict(
#         type='RepeatDataset',
#         times=2,
#         dataset=dict(
#             type=dataset_type,
#             ann_file=data_root + 'train.txt',
#             img_prefix=data_root + 'WIDER_train/',
#             min_size=17,
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'val.txt',
#         img_prefix=data_root + 'WIDER_val/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'val.txt',
#         img_prefix=data_root + 'WIDER_val/',
#         pipeline=test_pipeline))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_train/train.txt',
        img_prefix=data_root + 'WIDER_train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root +  'WIDER_val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=test_pipeline))

# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=dict(type='BN'),
        upsample_cfg=dict(mode='bilinear')
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        #anchor_generator = dict(
        #    type='AnchorGenerator',
        #    octave_base_scale=None,
        #    scales_per_octave=None,
        #    ratios=[1.0],
        #    strides=[8, 16, 32],
        #    base_sizes = [16, 64, 256],
        #    scales = [1.0, 2.0]),
        #anchor_generator=dict(
        #    type='AnchorGenerator',
        #    octave_base_scale=2 ** (4 / 3),
        #    scales_per_octave=3,
        #    ratios=[1.3],
        #    base_sizes= [4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma = 2.0,
            alpha = 0.25,
            loss_weight=1.0),
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        #loss_bbox=dict(
        #    type='DIoULoss',
        #    loss_weight=2.0)),

    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        # sampler=dict(
        #     type='OHEMSampler',
        #     num=512,
        #     pos_fraction=0.25,
        #     context = None,
        #     neg_pos_ub=-1,
        #     add_gt_as_proposals=True
        # ),
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))



# model = dict(
#     neck=dict(
#         # type='FPN',
#         # in_channels=[256, 512, 1024, 2048],
#         # out_channels=256,
#         # start_level=1,
#         # add_extra_convs='on_input',
#         num_outs=3),
#     bbox_head=dict(
#         num_classes=1,
#         anchor_generator = dict(
#             type='AnchorGenerator',
#             octave_base_scale=None,
#             scales_per_octave=None,
#             ratios=[1.0],
#             strides=[8, 16, 32],
#             base_sizes = [16, 64, 256],
#             scales = [1.0, 2.0]),
#         # loss_cls=dict(
#         #     gamma=0,
#         #     alpha=1.0,
#         # )
#     ),
#     # train_cfg=dict(
#     #     sampler=dict(
#     #         type='OHEMSampler',
#     #         num=1000,
#     #         pos_fraction=0.25,
#     #         neg_pos_ub=-1,
#     #         add_gt_as_proposals=True
#     #     )
#     # )
# )
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[16, 30, 40])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=70)
log_config = dict(interval=1)
#resume_from = '/raid/datazyp/Swin-Transformer-Object-Detection/work_dirs/retinanet_r50_fpn_widerface_640x640/latest.pth'