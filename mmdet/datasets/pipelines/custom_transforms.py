# import numpy as np
# import mmcv
# from .transforms import Resize
from ..builder import PIPELINES
import cv2
import numpy as np
import random
from .box_utils import matrix_iof

@PIPELINES.register_module()
class ShortSideResize(object):
    def __init__(self, imgdim):
        super(ShortSideResize, self).__init__()
        self.imgdim = imgdim

    def __call__(self, results):
        img = results['img']
        for key in results.get("bbox_fields", []):
            boxes = results[key]

        return results




def _crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        # landms_t = landm[mask_a].copy()
        # landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        # landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        # landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        # landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        # landms_t = landms_t.reshape([-1, 10])


        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        # landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        # landms = landms.copy()
        # landms = landms.reshape([-1, 5, 2])
        # landms[:, :, 0] = width - landms[:, :, 0]
        # tmp = landms[:, 1, :].copy()
        # landms[:, 1, :] = landms[:, 0, :]
        # landms[:, 0, :] = tmp
        # tmp1 = landms[:, 4, :].copy()
        # landms[:, 4, :] = landms[:, 3, :]
        # landms[:, 3, :] = tmp1
        # landms = landms.reshape([-1, 10])

    return image, boxes


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image


@PIPELINES.register_module()
class RetinafacePreprocess(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            #'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        #assert targets.shape[0] > 0, "this image does not have gt"

        image = results['img']
        #boxes =  np.empty((0, 4))
        #labels = np.empty((0,))
        for key in results.get('bbox_fields', []): #['gt_bboxes_ignore', 'gt_bboxes']
            if key == 'gt_bboxes_ignore':   # no considering the ignore situation TODO
                continue
            boxes = results[key]
            label_key = self.bbox2label.get(key)
            #if label_key in results:
            labels = results[label_key]


        # boxes = targets[:, :4].copy()
        # labels = targets[:, -1].copy()
        # #landm = targets[:, 4:-1].copy()

            image_t, boxes_t, labels_t, pad_image_flag = _crop(image, boxes, labels, self.img_dim)
            image_t = _distort(image_t)
            image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
            image_t, boxes_t = _mirror(image_t, boxes_t)
            height, width, _ = image_t.shape
            image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height

            # landm_t[:, 0::2] /= width
            # landm_t[:, 1::2] /= height

            labels_t = np.expand_dims(labels_t, 1)
            targets_t = np.hstack((boxes_t, labels_t))


            results[key] = boxes_t
            results['img'] = image_t

        return results


@PIPELINES.register_module()
class RandomSquareCrop(object):
    """
    Random crop the square patch of image & bboxes with a size from Args
    (crop_ratio_range or crop_choice) of the short edge of image and keep
    the overlapped part of box if its center is within the cropped patch.

    Args:
        crop_ratio_range (list): a list of two elements (min, max)
        crop_choice (list): a list of crop ratio.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self, crop_ratio_range=None, crop_choice=None):

        self.crop_ratio_range = crop_ratio_range
        self.crop_choice = crop_choice

        assert (self.crop_ratio_range is None) ^ (self.crop_choice is None)
        if self.crop_ratio_range is not None:
            self.crop_ratio_min, self.crop_ratio_max = self.crop_ratio_range

        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape

        # while True:

        if self.crop_ratio_range is not None:
            scale = np.random.uniform(
                self.crop_ratio_min,
                self.crop_ratio_max,
            )
        elif self.crop_choice is not None:
            scale = np.random.choice(self.crop_choice)

        for i in range(250):
            short_side = min(w, h)
            cw = int(scale * short_side)
            ch = cw

            # TODO +1
            left = random.uniform(0, w - cw)
            top = random.uniform(0, h - ch)

            patch = np.array(
                (int(left), int(top), int(left + cw), int(top + ch)))

            # center of boxes should inside the crop img
            # only adjust boxes and instance masks when the gt is not empty
            # adjust boxes
            def is_center_of_bboxes_in_patch(boxes, patch):
                # TODO >=
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) *
                        (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) *
                        (center[:, 1] < patch[3]))
                return mask

            mask = is_center_of_bboxes_in_patch(boxes, patch)
            if not mask.any() and not i == 249:
                continue
            for key in results.get('bbox_fields', []):
                boxes = results[key].copy()
                mask = is_center_of_bboxes_in_patch(boxes, patch)
                boxes = boxes[mask]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results[key] = boxes
                # labels
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][mask]

                # mask fields
                mask_key = self.bbox2mask.get(key)
                if mask_key in results:
                    ind = mask.nonzero(as_tuple=False)[0]
                    results[mask_key] = results[mask_key][ind].crop(patch)

            # adjust the img no matter whether the gt is empty before crop
            img = img[patch[1]:patch[3], patch[0]:patch[2]]
            results['img'] = img
            results['img_shape'] = img.shape

            # seg fields
            for key in results.get('seg_fields', []):
                results[key] = results[key][patch[1]:patch[3],
                                            patch[0]:patch[2]]
            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_iou}, '
        repr_str += f'crop_size={self.crop_size})'
        return repr_str
