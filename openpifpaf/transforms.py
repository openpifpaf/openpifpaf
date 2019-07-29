"""Transform input data.

Images are resized with Pillow which has a different coordinate convention:
https://pillow.readthedocs.io/en/3.3.x/handbook/concepts.html#coordinate-system

> The Python Imaging Library uses a Cartesian pixel coordinate system,
  with (0,0) in the upper left corner. Note that the coordinates refer to
  the implied pixel corners; the centre of a pixel addressed as (0, 0)
  actually lies at (0.5, 0.5).
"""

from abc import ABCMeta, abstractmethod
import copy
import io
import logging
import math
import numpy as np
import PIL
import scipy
import torch
import torchvision

from .utils import horizontal_swap_coco


class Preprocess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image, anns, meta):
        """Implementation of preprocess operation."""

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets = keypoint_sets.copy()

        keypoint_sets[:, :, 0] += meta['offset'][0]
        keypoint_sets[:, :, 1] += meta['offset'][1]

        keypoint_sets[:, :, 0] = (keypoint_sets[:, :, 0] + 0.5) / meta['scale'][0] - 0.5
        keypoint_sets[:, :, 1] = (keypoint_sets[:, :, 1] + 0.5) / meta['scale'][1] - 0.5

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] - 1.0 + w
            for keypoints in keypoint_sets:
                if meta.get('horizontal_swap'):
                    keypoints[:] = meta['horizontal_swap'](keypoints)

        return keypoint_sets


class ImageTransform(Preprocess):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, image, anns, meta):
        image = self.image_transform(image)
        return image, anns, meta


class NormalizeAnnotations(Preprocess):
    @staticmethod
    def normalize_annotations(anns):
        anns = copy.deepcopy(anns)

        for ann in anns:
            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            ann['bbox_original'] = np.copy(ann['bbox'])
            if 'segementation' in ann:
                del ann['segmentation']

        return anns

    def __call__(self, image, anns, meta):
        anns = self.normalize_annotations(anns)

        if meta is None:
            w, h = image.size
            meta = {
                'offset': np.array((0.0, 0.0)),
                'scale': np.array((1.0, 1.0)),
                'valid_area': np.array((0.0, 0.0, w, h)),
                'hflip': False,
                'width_height': np.array((w, h)),
            }

        return image, anns, meta


class Compose(Preprocess):
    def __init__(self, preprocess_list):
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta):
        for p in self.preprocess_list:
            image, anns, meta = p(image, anns, meta)

        return image, anns, meta


class MultiScale(Preprocess):
    def __init__(self, preprocess_list):
        """Create lists of preprocesses.

        Must be the most outer preprocess function.
        Preprocess_list can contain transforms.Compose() functions.
        """
        self.preprocess_list = preprocess_list

    def __call__(self, image, anns, meta):
        image_list, anns_list, meta_list = [], [], []
        for p in self.preprocess_list:
            this_image, this_anns, this_meta = p(image, anns, meta)
            image_list.append(this_image)
            anns_list.append(this_anns)
            meta_list.append(this_meta)

        return image_list, anns_list, meta_list


class RescaleRelative(Preprocess):
    def __init__(self, scale_range=(0.5, 1.0), *, resample=PIL.Image.BICUBIC):
        self.log = logging.getLogger(self.__class__.__name__)
        self.scale_range = scale_range
        self.resample = resample

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        if isinstance(self.scale_range, tuple):
            scale_factor = (
                self.scale_range[0] +
                torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0])
            )
        else:
            scale_factor = self.scale_range

        image, anns, scale_factors = self.scale(image, anns, scale_factor)
        self.log.debug('meta before: %s', meta)
        meta['offset'] *= scale_factors
        meta['scale'] *= scale_factors
        meta['valid_area'][:2] *= scale_factors
        meta['valid_area'][2:] *= scale_factors
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def scale(self, image, anns, factor):
        # scale image
        w, h = image.size
        image = image.resize((int(w * factor), int(h * factor)), self.resample)
        self.log.debug('before resize = (%f, %f), after = %s', w, h, image.size)

        # rescale keypoints
        x_scale = image.size[0] / w
        y_scale = image.size[1] / h
        for ann in anns:
            ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
            ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale

        return image, anns, np.array((x_scale, y_scale))


class RescaleAbsolute(Preprocess):
    def __init__(self, long_edge, *, resample=PIL.Image.BICUBIC):
        self.log = logging.getLogger(self.__class__.__name__)
        self.long_edge = long_edge
        self.resample = resample

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, scale_factors = self.scale(image, anns)
        self.log.debug('meta before: %s', meta)
        meta['offset'] *= scale_factors
        meta['scale'] *= scale_factors
        meta['valid_area'][:2] *= scale_factors
        meta['valid_area'][2:] *= scale_factors
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def scale(self, image, anns):
        # scale image
        w, h = image.size

        this_long_edge = self.long_edge
        if isinstance(this_long_edge, (tuple, list)):
            this_long_edge = int(torch.randint(this_long_edge[0], this_long_edge[1], (1,)).item())

        s = this_long_edge / max(h, w)
        if h > w:
            image = image.resize((int(w * s), this_long_edge), self.resample)
        else:
            image = image.resize((this_long_edge, int(h * s)), self.resample)
        self.log.debug('before resize = (%f, %f), scale factor = %f, after = %s',
                       w, h, s, image.size)

        # rescale keypoints
        x_scale = image.size[0] / w
        y_scale = image.size[1] / h
        for ann in anns:
            ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
            ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale

        return image, anns, np.array((x_scale, y_scale))


class Crop(Preprocess):
    def __init__(self, long_edge):
        self.log = logging.getLogger(self.__class__.__name__)
        self.long_edge = long_edge

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.crop(image, anns)
        meta['offset'] += ltrb[:2]

        self.log.debug('valid area before crop of %s: %s', ltrb, meta['valid_area'])
        # process crops from left and top
        meta['valid_area'][:2] = np.maximum(0.0, meta['valid_area'][:2] - ltrb[:2])
        meta['valid_area'][2:] = np.maximum(0.0, meta['valid_area'][2:] - ltrb[:2])
        # process cropps from right and bottom
        meta['valid_area'][2:] = np.minimum(meta['valid_area'][2:], ltrb[2:] - ltrb[:2])
        self.log.debug('valid area after crop: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def crop(self, image, anns):
        w, h = image.size
        padding = int(self.long_edge / 2.0)
        x_offset, y_offset = 0, 0
        if w > self.long_edge:
            x_offset = torch.randint(-padding, w - self.long_edge + padding, (1,))
            x_offset = torch.clamp(x_offset, min=0, max=w - self.long_edge).item()
        if h > self.long_edge:
            y_offset = torch.randint(-padding, h - self.long_edge + padding, (1,))
            y_offset = torch.clamp(y_offset, min=0, max=h - self.long_edge).item()
        self.log.debug('crop offsets (%d, %d)', x_offset, y_offset)

        # crop image
        new_w = min(self.long_edge, w - x_offset)
        new_h = min(self.long_edge, h - y_offset)
        ltrb = (x_offset, y_offset, x_offset + new_w, y_offset + new_h)
        image = image.crop(ltrb)

        # crop keypoints
        for ann in anns:
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset

        return image, anns, np.array(ltrb)


class CenterPad(Preprocess):
    def __init__(self, target_size):
        self.log = logging.getLogger(self.__class__.__name__)

        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]

        self.log.debug('valid area before pad with %s: %s', ltrb, meta['valid_area'])
        meta['valid_area'][:2] += ltrb[:2]
        self.log.debug('valid area after pad: %s', meta['valid_area'])

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    def center_pad(self, image, anns):
        w, h = image.size
        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        ltrb = (
            left,
            top,
            self.target_size[0] - w - left,
            self.target_size[1] - h - top,
        )

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb


class HFlip(Preprocess):
    def __init__(self, *, swap=horizontal_swap_coco):
        self.swap = swap

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w
        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta


class RandomApply(Preprocess):
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, image, anns, meta):
        if float(torch.rand(1).item()) > self.probability:
            return image, anns, meta
        return self.transform(image, anns, meta)


class RotateBy90(Preprocess):
    def __init__(self, angle_perturbation=5.0):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)

        self.angle_perturbation = angle_perturbation

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, h = image.size
        rnd1 = float(torch.rand(1).item())
        angle = int(rnd1 * 4.0) * 90.0
        rnd2 = float(torch.rand(1).item())
        angle += -self.angle_perturbation + 2.0 * self.angle_perturbation * rnd2
        self.log.debug('rotation angle = %f', angle)

        # rotate image
        im_np = np.asarray(image)
        im_np = scipy.ndimage.rotate(im_np, angle=angle, cval=127, reshape=False)
        image = PIL.Image.fromarray(im_np)
        self.log.debug('rotated by = %f degrees', angle)

        # rotate keypoints
        cangle = math.cos(angle / 180.0 * math.pi)
        sangle = math.sin(angle / 180.0 * math.pi)
        for ann in anns:
            xy = ann['keypoints'][:, :2]
            x_old = xy[:, 0].copy() - w/2
            y_old = xy[:, 1].copy() - h/2
            xy[:, 0] = w/2 + cangle * x_old + sangle * y_old
            xy[:, 1] = h/2 - sangle * x_old + cangle * y_old
            ann['bbox'] = self.rotate_box(ann['bbox'], w, h, angle)

        self.log.debug('meta before: %s', meta)
        meta['valid_area'] = self.rotate_box(meta['valid_area'], w, h, angle)
        self.log.debug('meta after: %s', meta)

        for ann in anns:
            ann['valid_area'] = meta['valid_area']

        return image, anns, meta

    @staticmethod
    def rotate_box(bbox, width, height, angle_degrees):
        """Input bounding box is of the form x, y, width, height."""

        cangle = math.cos(angle_degrees / 180.0 * math.pi)
        sangle = math.sin(angle_degrees / 180.0 * math.pi)

        four_corners = np.array([
            [bbox[0], bbox[1]],
            [bbox[0] + bbox[2], bbox[1]],
            [bbox[0], bbox[1] + bbox[3]],
            [bbox[0] + bbox[2], bbox[1] + bbox[3]],
        ])

        x_old = four_corners[:, 0].copy() - width/2
        y_old = four_corners[:, 1].copy() - height/2
        four_corners[:, 0] = width/2 + cangle * x_old + sangle * y_old
        four_corners[:, 1] = height/2 - sangle * x_old + cangle * y_old

        x = np.min(four_corners[:, 0])
        y = np.min(four_corners[:, 1])
        xmax = np.max(four_corners[:, 0])
        ymax = np.max(four_corners[:, 1])

        return np.array([x, y, xmax - x, ymax - y])


class JpegCompression(Preprocess):
    def __init__(self, quality=50):
        self.quality = quality

    def __call__(self, image, anns, meta):
        f = io.BytesIO()
        image.save(f, 'jpeg', quality=self.quality)
        return PIL.Image.open(f), anns, meta


class Blur(Preprocess):
    def __init__(self, max_sigma=5.0):
        self.max_sigma = max_sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.max_sigma * float(torch.rand(1).item())
        im_np = scipy.ndimage.filters.gaussian_filter(im_np, sigma=(sigma, sigma, 0))
        return PIL.Image.fromarray(im_np), anns, meta


EVAL_TRANSFORM = Compose([
    NormalizeAnnotations(),
    ImageTransform(torchvision.transforms.ToTensor()),
    ImageTransform(
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ),
])


TRAIN_TRANSFORM = Compose([
    NormalizeAnnotations(),
    ImageTransform(torchvision.transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)),
    RandomApply(JpegCompression(), 0.1),  # maybe irrelevant for COCO, but good for others
    # RandomApply(Blur(), 0.01),  # maybe irrelevant for COCO, but good for others
    ImageTransform(torchvision.transforms.RandomGrayscale(p=0.01)),
    EVAL_TRANSFORM,
])
