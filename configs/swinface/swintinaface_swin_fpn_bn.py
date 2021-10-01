# should include default runtime
_base_ = [
    '../_base_/default_runtime.py'
]

# 1. data
dataset_type = 'WIDERFaceDataset'
data_root = '/raid/datazyp/DATASET/WIDER_voc2/' # 'data/WIDERFace/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
size_divisor = 32

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_train/train.txt',
        img_prefix=data_root + 'WIDER_train/',
        min_size=1,
        #offset=0, #TODO offset == 1 in mmdetection, to check
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomSquareCrop',
                 crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
            dict(
                type='PhotoMetricDistortionRe', # Re fix the pixel value not in range 0 - 255, limit the range
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes',
                                           'gt_labels', 'gt_bboxes_ignore']),
        ]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        #min_size=1,
        #offset=0,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32, pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        # min_size=1,
        # offset=0,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32, pad_val=0),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
)

# 2. model
num_classes = 1
strides = [4, 8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [1.3]
#num_anchors = scales_per_octave * len(ratios)

anchor_generator = dict(
    type='AnchorGenerator',
    octave_base_scale=2**(4 / 3),
    scales_per_octave=scales_per_octave,
    ratios=ratios,
    strides=strides,
    base_sizes=strides)

bbox_coder = dict(
    type='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[0.1, 0.1, 0.2, 0.2])

model = dict(
    type='SingleStageDetector',
    pretrained= None, # 'torchvision://resnet50',
    backbone=dict(
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
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),

    # neck=
    #     dict(
    #         type='FPN',
    #         in_channels=[256, 512, 1024, 2048],
    #         out_channels=256,
    #         start_level=0,
    #         add_extra_convs='on_input',
    #         num_outs=6,
    #         norm_cfg=dict(type='BN'),
    #         upsample_cfg=dict(mode='bilinear')
    #     ),

    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6,
        norm_cfg=dict(type='BN'),
        upsample_cfg=dict(mode='bilinear')),

        # TODO add Inception block implementation
        # dict(
        #     type='Inception',
        #     in_channel=256,
        #     num_levels=6,
        #     norm_cfg=dict(type='BN'),
        #     share=True)

    # head=dict(
    #     type='IoUAwareRetinaHead',
    #     num_classes=num_classes,
    #     num_anchors=num_anchors,
    #     in_channels=256,
    #     stacked_convs=4,
    #     feat_channels=256,
    #     norm_cfg=dict(type='BN'),
    #     use_sigmoid=use_sigmoid))
    bbox_head =
        dict(
            type='RetinaHead',
            num_classes=1,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=anchor_generator,
            norm_cfg=dict(type='BN'),
            # no use sigmoid, in loss cls

            bbox_coder=bbox_coder,
            reg_decoded_bbox = True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma = 2.0,
                alpha = 0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='DIoULoss', loss_weight=2.0)
        ),

    # TODO add loss iou in tinaface
    train_cfg = dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.35,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1,
            gpu_assign_thr=100
        ),

        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),

    # TODO check test cfg, no IoUBBoxAnchorConverter in mmdetection
    test_cfg=dict(
        nms_pre=-1,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=-1
    )

)




# 3. engines
# meshgrid = dict(
#     type='BBoxAnchorMeshGrid',
#     strides=strides,
#     base_anchor=dict(
#         type='BBoxBaseAnchor',
#         octave_base_scale=2**(4 / 3),
#         scales_per_octave=scales_per_octave,
#         ratios=ratios,
#         base_sizes=strides))



# train_engine = dict(
#     type='TrainEngine',
#     model=model,
#     criterion=dict(
#         type='IoUBBoxAnchorCriterion',
#         num_classes=num_classes,
#         meshgrid=meshgrid,
#         bbox_coder=bbox_coder,
#         loss_cls=dict(
#             type='FocalLoss',
#             use_sigmoid=use_sigmoid,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0),
#         reg_decoded_bbox=True,
#         loss_bbox=dict(type='DIoULoss', loss_weight=2.0),
#         loss_iou=dict(
#             type='CrossEntropyLoss',
#             use_sigmoid=True,
#             loss_weight=1.0),
#         train_cfg=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.35,
#                 neg_iou_thr=0.35,
#                 min_pos_iou=0.35,
#                 ignore_iof_thr=-1,
#                 gpu_assign_thr=100),
#             allowed_border=-1,
#             pos_weight=-1,
#             debug=False)),
#     optimizer=dict(type='SGD', lr=3.75e-3, momentum=0.9, weight_decay=5e-4)) # 3 GPUS

evaluation = dict(interval=30)

optimizer = dict(type='SGD', lr=3.75e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# ## 3.2 val engine
# val_engine = dict(
#     type='ValEngine',
#     model=model,
#     meshgrid=meshgrid,
#     converter=dict(
#         type='IoUBBoxAnchorConverter',
#         num_classes=num_classes,
#         bbox_coder=bbox_coder,
#         nms_pre=-1,
#         use_sigmoid=use_sigmoid),
#     num_classes=num_classes,
#     test_cfg=dict(
#         min_bbox_size=0,
#         score_thr=0.01,
#         nms=dict(type='lb_nms', iou_thr=0.45),
#         max_per_img=-1),
#     use_sigmoid=use_sigmoid,
#     eval_metric=None)
lr_config = dict(
    policy='CosineRestart',
    periods=[30] * 21,
    restart_weights=[1] * 21,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-1,
    min_lr_ratio=1e-2)


# hooks = [
#     dict(type='OptimizerHook'), # TODO no OptimizerHook, only DistOptimizerHook, to check
#     dict(
#         type='CosineRestartLrUpdaterHook',
#         periods=[30] * 21,
#         restart_weights=[1] * 21,
#         warmup='linear',
#         warmup_iters=500,
#         warmup_ratio=1e-1,
#         min_lr_ratio=1e-2),
#     dict(type='EvalHook'),
#     dict(type='SnapshotHook', interval=1),
#     dict(type='LoggerHook', interval=100)
# ]
# evaluation = dict(  # The config to build the evaluation hook, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7 for more details.
#     interval=1,  # Evaluation interval
#     metric=['bbox'])
# 5. work modes
#modes = ['train']#, 'val']
max_epochs = 630

# 6. checkpoint
# weights = dict(
#     filepath='torchvision://resnet50',
#     prefix='backbone')
# optimizer = dict(filepath='workdir/retinanet_mini/epoch_3_optim.pth')
# meta = dict(filepath='workdir/retinanet_mini/epoch_3_meta.pth')



# runner: necessary for mmdetection
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(
    interval=1)  # The save interval is 1

# 7. misc
#seed = 1234
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = '/raid/datazyp/Swin-Transformer-Object-Detection/work_dir_swinface_reim/latest.pth' #None

workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = 'work_dir_swinface_reim'
