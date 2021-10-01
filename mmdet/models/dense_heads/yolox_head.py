#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

# from yolox.utils import bboxes_iou

# from .losses import IOUloss
# from .network_blocks import BaseConv, DWConv

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

@HEADS.register_module()
class YOLOXHead(BaseDenseHead, BBoxTestMixin):
    def __init__(
        # self,
        # num_classes,
        # width=1.0,
        # strides=[8, 16, 32],
        # in_channels=[256, 512, 1024],
        # act="lrelu",
        # depthwise=False,
        self,
        num_classes,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=None,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        width = 1.0,
        act="lrelu",
        depthwise = False,
        **kwargs
    ):

        """
        Args:
            act (str): activation type of conv. Default value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Default value: False.
        """
        super().__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)

        #self.n_anchors = 1
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.width = width
        self.act = act
        self.decode_in_inference = True  # for deploy, set to False


    # def initialize_biases(self, prior_prob):    # TODO check initialization process
    #     for conv in self.cls_preds:
    #         b = conv.bias.view(self.n_anchors, -1)
    #         b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
    #         conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    #
    #     for conv in self.obj_preds:
    #         b = conv.bias.view(self.n_anchors, -1)
    #         b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
    #         conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = BaseConv #DWConv if depthwise else BaseConv

        #3for i in range(len(in_channels)):
        self.stems.append(
            BaseConv(
                in_channels=int(self.in_channels * self.width),
                out_channels=int(256 * self.width),
                ksize=1,
                stride=1,
                act=self.act,
            )
        )
        self.cls_convs.append(
            nn.Sequential(
                *[
                    Conv(
                        in_channels=int(256 * self.width),
                        out_channels=int(256 * self.width),
                        ksize=3,
                        stride=1,
                        act=self.act,
                    ),
                    Conv(
                        in_channels=int(256 * self.width),
                        out_channels=int(256 * self.width),
                        ksize=3,
                        stride=1,
                        act=self.act,
                    ),
                ]
            )
        )
        self.reg_convs.append(
            nn.Sequential(
                *[
                    Conv(
                        in_channels=int(256 * self.width),
                        out_channels=int(256 * self.width),
                        ksize=3,
                        stride=1,
                        act=self.act,
                    ),
                    Conv(
                        in_channels=int(256 * self.width),
                        out_channels=int(256 * self.width),
                        ksize=3,
                        stride=1,
                        act=self.act,
                    ),
                ]
            )
        )
        self.cls_preds.append(
            nn.Conv2d(
                in_channels=int(256 * self.width),
                out_channels=self.n_anchors * self.num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.reg_preds.append(
            nn.Conv2d(
                in_channels=int(256 * self.width),
                out_channels=4,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.obj_preds.append(
            nn.Conv2d(
                in_channels=int(256 * self.width),
                out_channels=self.n_anchors * 1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )


    def forward_single(self, x):
        # cls_feat = x
        # reg_feat = x
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)
        # cls_score = self.retina_cls(cls_feat)
        # bbox_pred = self.retina_reg(reg_feat)
        # return cls_score, bbox_pred
        x = self.stems(x)
        cls_x = x
        reg_x = x

        cls_feat = self.cls_conv(cls_x)
        cls_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_conv(reg_x)
        reg_output = self.reg_preds(reg_feat)

        obj_output = self.obj_preds(reg_feat)

        return cls_output, reg_output, obj_output

    # def forward(self, xin, labels=None, imgs=None):
    #     outputs = []
    #     origin_preds = []
    #     x_shifts = []
    #     y_shifts = []
    #     expanded_strides = []
    #
    #     for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
    #         zip(self.cls_convs, self.reg_convs, self.strides, xin)
    #     ):
    #         x = self.stems[k](x)
    #         cls_x = x
    #         reg_x = x
    #
    #         cls_feat = cls_conv(cls_x)
    #         cls_output = self.cls_preds[k](cls_feat)
    #
    #         reg_feat = reg_conv(reg_x)
    #         reg_output = self.reg_preds[k](reg_feat)
    #         obj_output = self.obj_preds[k](reg_feat)
    #
    #         if self.training:
    #             output = torch.cat([reg_output, obj_output, cls_output], 1)
    #             output, grid = self.get_output_and_grid(
    #                 output, k, stride_this_level, xin[0].type()
    #             )
    #             x_shifts.append(grid[:, :, 0])
    #             y_shifts.append(grid[:, :, 1])
    #             expanded_strides.append(
    #                 torch.zeros(1, grid.shape[1])
    #                 .fill_(stride_this_level)
    #                 .type_as(xin[0])
    #             )
    #             if self.use_l1:
    #                 batch_size = reg_output.shape[0]
    #                 hsize, wsize = reg_output.shape[-2:]
    #                 reg_output = reg_output.view(
    #                     batch_size, self.n_anchors, 4, hsize, wsize
    #                 )
    #                 reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
    #                     batch_size, -1, 4
    #                 )
    #                 origin_preds.append(reg_output.clone())
    #
    #         else:
    #             output = torch.cat(
    #                 [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
    #             )
    #
    #
    #         self.hw = [x.shape[-2:] for x in outputs]
    #         # [batch, n_anchors_all, 85]
    #         outputs = torch.cat(
    #             [x.flatten(start_dim=2) for x in outputs], dim=2
    #         ).permute(0, 2, 1)
    #         if self.decode_in_inference:
    #             return self.decode_outputs(outputs, dtype=xin[0].type())
    #         else:
    #             return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # # classification loss
        # labels = labels.reshape(-1)
        # label_weights = label_weights.reshape(-1)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)
        # # regression loss
        # bbox_targets = bbox_targets.reshape(-1, 4)
        # bbox_weights = bbox_weights.reshape(-1, 4)
        # bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        # if self.reg_decoded_bbox:
        #     # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
        #     # is applied directly on the decoded bounding boxes, it
        #     # decodes the already encoded coordinates to absolute format.
        #     anchors = anchors.reshape(-1, 4)
        #     bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     avg_factor=num_total_samples)

        # bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        # obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        # cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        bs, num_anchors, h, w = cls_score.shape
        bbox_preds = bbox_pred.reshape(0, -1, 4, 0, 0).permute(0, 1, 3, 4, 2) # TODO multi class situation
        bbox_preds = bbox_preds.reshape(bs, -1, 4)

        cls_preds = cls_score.reshape(0, -1, 1, 0, 0).permute(0, 1, 3, 4, 2)  # TODO multi class situation
        cls_preds = bbox_preds.reshape(bs, -1, 1)

        # calculate targets
        # mixup = labels.shape[2] > 5
        # if mixup:
        #     label_cut = labels[..., :5]
        # else:
        #     label_cut = labels
        #nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        # TODO calculate num of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        # return (
        #     loss,
        #     reg_weight * loss_iou,
        #     loss_obj,
        #     loss_cls,
        #     loss_l1,
        #     num_fg / max(num_gts, 1),
        # )


        return loss_cls, loss_bbox

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def loss(self,
    #          cls_scores,
    #          bbox_preds,
    #          gt_bboxes,
    #          gt_labels,
    #          img_metas,
    #          gt_bboxes_ignore=None):
    #     """Compute losses of the head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W)
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W)
    #         gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
    #             shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): class indices corresponding to each box
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         gt_bboxes_ignore (None | list[Tensor]): specify which bounding
    #             boxes can be ignored when computing the loss. Default: None
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    #     assert len(featmap_sizes) == self.anchor_generator.num_levels
    #
    #     device = cls_scores[0].device
    #
    #     anchor_list, valid_flag_list = self.get_anchors(
    #         featmap_sizes, img_metas, device=device)
    #     label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
    #     cls_reg_targets = self.get_targets(
    #         anchor_list,
    #         valid_flag_list,
    #         gt_bboxes,
    #         img_metas,
    #         gt_bboxes_ignore_list=gt_bboxes_ignore,
    #         gt_labels_list=gt_labels,
    #         label_channels=label_channels)
    #     if cls_reg_targets is None:
    #         return None
    #     (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
    #      num_total_pos, num_total_neg) = cls_reg_targets
    #     num_total_samples = (
    #         num_total_pos + num_total_neg if self.sampling else num_total_pos)
    #
    #     # anchor number of multi levels
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #     # concat all level anchors and flags to a single tensor
    #     concat_anchor_list = []
    #     for i in range(len(anchor_list)):
    #         concat_anchor_list.append(torch.cat(anchor_list[i]))
    #     all_anchor_list = images_to_levels(concat_anchor_list,
    #                                        num_level_anchors)
    #
    #     losses_cls, losses_bbox = multi_apply(
    #         self.loss_single,
    #         cls_scores,
    #         bbox_preds,
    #         all_anchor_list,
    #         labels_list,
    #         label_weights_list,
    #         bbox_targets_list,
    #         bbox_weights_list,
    #         num_total_samples=num_total_samples)
    #     return dict(loss_cls=losses_cls, loss_bbox=losses_bbox) # 返回的字典后续是怎么操作的？

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        '''


        imgs: input images, not necessary
        x_shifts: x dim shifts of each anchor [1, n_anchors_all], get from ...
        y_shifts: y dim shifts of each anchor [1, n_anchors_all]
        expanded_strides: to check
        labels: labels in origin format, before anchor assignment, [bs, num_gts, 5] , last dim 0, class label, 1:5, reg label
        outputs: output in format []
        origin_preds: only used when use l1 is True
        dtype: used for object score dtype??
        '''
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
