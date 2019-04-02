"""Transform input data.

Images are resized with Pillow which has a different coordinate convention:
https://pillow.readthedocs.io/en/3.3.x/handbook/concepts.html#coordinate-system

> The Python Imaging Library uses a Cartesian pixel coordinate system,
  with (0,0) in the upper left corner. Note that the coordinates refer to
  the implied pixel corners; the centre of a pixel addressed as (0, 0)
  actually lies at (0.5, 0.5).
"""

import io
import numpy as np
import PIL
import torch
import torchvision

from .utils import horizontal_swap


def jpeg_compression_augmentation(im):
    f = io.BytesIO()
    im.save(f, 'jpeg', quality=50)
    return PIL.Image.open(f)


normalize = torchvision.transforms.Normalize(  # pylint: disable=invalid-name
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


image_transform = torchvision.transforms.Compose([  # pylint: disable=invalid-name
    torchvision.transforms.ToTensor(),
    normalize,
])


image_transform_train = torchvision.transforms.Compose([  # pylint: disable=invalid-name
    torchvision.transforms.ColorJitter(brightness=0.1,
                                       contrast=0.1,
                                       saturation=0.1,
                                       hue=0.1),
    torchvision.transforms.RandomApply([
        # maybe not relevant for COCO, but good for other datasets:
        torchvision.transforms.Lambda(jpeg_compression_augmentation),
    ], p=0.1),
    torchvision.transforms.RandomGrayscale(p=0.01),
    torchvision.transforms.ToTensor(),
    normalize,
])


class Preprocess(object):
    @staticmethod
    def normalize_annotations(anns):
        # convert as much data as possible to numpy arrays to avoid every float
        # being turned into its own torch.Tensor()
        for ann in anns:
            ann['keypoints'] = np.asarray(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            ann['bbox'] = np.asarray(ann['bbox'], dtype=np.float32)
            ann['bbox_original'] = np.copy(ann['bbox'])
            del ann['segmentation']

        return anns

    def __call__(self, image, anns):
        w, h = image.size
        anns = self.normalize_annotations(anns)

        meta = {
            'offset': (0.0, 0.0),
            'scale': (1.0, 1.0),
            'valid_area': (0.0, 0.0, w, h),
            'hflip': False,
            'width_height': (w, h),
        }
        return image, anns, meta


class SquareRescale(object):
    def __init__(self, long_edge, *,
                 black_bars=False, random_hflip=False,
                 normalize_annotations=Preprocess.normalize_annotations):
        self.long_edge = long_edge
        self.black_bars = black_bars
        self.random_hflip = random_hflip
        self.normalize_annotations = normalize_annotations

    def scale_long_edge(self, image):
        w, h = image.size
        s = self.long_edge / max(h, w)
        if h > w:
            return torchvision.transforms.functional.resize(
                image, (self.long_edge, int(w * s)), PIL.Image.BICUBIC)
        return torchvision.transforms.functional.resize(
            image, (int(h * s), self.long_edge), PIL.Image.BICUBIC)

    def center_pad(self, image):
        w, h = image.size
        left = int((self.long_edge - w) / 2.0)
        top = int((self.long_edge - h) / 2.0)
        ltrb = (
            left,
            top,
            self.long_edge - w - left,
            self.long_edge - h - top,
        )
        return torchvision.transforms.functional.pad(
            image, ltrb, fill=(124, 116, 104))

    def __call__(self, image, anns):
        w, h = image.size
        if self.normalize_annotations is not None:
            anns = self.normalize_annotations(anns)

        # horizontal flip
        hflip = self.random_hflip and torch.rand(1).item() < 0.5
        if hflip:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for ann in anns:
                ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
                ann['keypoints'] = horizontal_swap(ann['keypoints'])  # TODO: hard coded for persons
                ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        image = self.scale_long_edge(image)
        if self.black_bars:
            image = self.center_pad(image)

        # rescale keypoints
        s = self.long_edge / max(h, w)
        w_rescaled, h_rescaled = int(w * s), int(h * s)
        x_scale = w_rescaled / w
        y_scale = h_rescaled / h
        for ann in anns:
            ann['keypoints'][:, 0] = (ann['keypoints'][:, 0] + 0.5) * x_scale - 0.5
            ann['keypoints'][:, 1] = (ann['keypoints'][:, 1] + 0.5) * y_scale - 0.5
            ann['bbox'][0] *= x_scale
            ann['bbox'][1] *= y_scale
            ann['bbox'][2] *= x_scale
            ann['bbox'][3] *= y_scale
            ann['scale'] = (x_scale, y_scale)

        # shift keypoints to center (like padding)
        if self.black_bars:
            x_offset = int((self.long_edge - w_rescaled) / 2.0)
            y_offset = int((self.long_edge - h_rescaled) / 2.0)
        else:
            x_offset, y_offset = 0, 0
        for ann in anns:
            ann['keypoints'][:, 0] += x_offset
            ann['keypoints'][:, 1] += y_offset
            ann['bbox'][0] += x_offset
            ann['bbox'][1] += y_offset
            ann['offset'] = (x_offset, y_offset)
            ann['valid_area'] = (x_offset, y_offset, w_rescaled, h_rescaled)

        meta = {
            'offset': (x_offset, y_offset),
            'scale': (x_scale, y_scale),
            'valid_area': (x_offset, y_offset, w_rescaled, h_rescaled),
            'hflip': hflip,
            'width_height': (w, h),
        }
        return image, anns, meta

    @staticmethod
    def keypoint_sets_inverse(keypoint_sets, meta):
        keypoint_sets[:, :, 0] -= meta['offset'][0]
        keypoint_sets[:, :, 1] -= meta['offset'][1]

        keypoint_sets[:, :, 0] = (keypoint_sets[:, :, 0] + 0.5) / meta['scale'][0] - 0.5
        keypoint_sets[:, :, 1] = (keypoint_sets[:, :, 1] + 0.5) / meta['scale'][1] - 0.5

        if meta['hflip']:
            w = meta['width_height'][0]
            keypoint_sets[:, :, 0] = -keypoint_sets[:, :, 0] - 1.0 + w
            for keypoints in keypoint_sets:
                keypoints[:] = horizontal_swap(keypoints)  # TODO: hard coded for persons

        return keypoint_sets


class SquareCrop(object):
    def __init__(self, edge, *,
                 min_scale=0.95, random_hflip=False,
                 normalize_annotations=Preprocess.normalize_annotations):
        self.target_edge = edge
        self.min_scale = min_scale
        self.random_hflip = random_hflip
        self.normalize_annotations = normalize_annotations

        self.image_resize = torchvision.transforms.Resize((edge, edge), PIL.Image.BICUBIC)

    def __call__(self, image, anns):
        w, h = image.size
        if self.normalize_annotations is not None:
            anns = self.normalize_annotations(anns)

        # horizontal flip
        hflip = self.random_hflip and torch.rand(1).item() < 0.5
        if hflip:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            for ann in anns:
                ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
                ann['keypoints'] = horizontal_swap(ann['keypoints'])  # TODO: hard coded for persons
                ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        # crop image
        short_edge = min(w, h)
        min_edge = int(short_edge * self.min_scale)
        if min_edge < short_edge:
            edge = int(torch.randint(min_edge, short_edge, (1,)).item())
        else:
            edge = short_edge
        # find crop offset
        padding = int(edge / 2.0)
        x_offset = torch.randint(-padding, w - edge + padding, (1,))
        x_offset = torch.clamp(x_offset, min=0, max=w - edge).item()
        y_offset = torch.randint(-padding, h - edge + padding, (1,))
        y_offset = torch.clamp(y_offset, min=0, max=h - edge).item()
        # print('crop offsets', x_offset, y_offset)
        image = image.crop((x_offset, y_offset, x_offset + edge, y_offset+edge))
        # print('square cropped image size', image.size)
        assert image.size[0] == image.size[1]

        # resize image
        image = self.image_resize(image)
        assert image.size[0] == image.size[1]
        assert image.size[0] == self.target_edge

        # annotations
        for ann in anns:
            # crop keypoints
            ann['keypoints'][:, 0] -= x_offset
            ann['keypoints'][:, 1] -= y_offset
            # resize keypoints
            ann['keypoints'][:, :2] = (
                (ann['keypoints'][:, :2] + 0.5) *
                self.target_edge / edge - 0.5
            )

            # bounding box
            ann['bbox'][0] -= x_offset
            ann['bbox'][1] -= y_offset
            ann['bbox'] *= self.target_edge / edge

            # valid area
            ann['valid_area'] = (0, 0, self.target_edge, self.target_edge)

        meta = {
            'offset': (x_offset, y_offset),
            # 'scale': (x_scale, y_scale),
            'scale': (0.0, 0.0),
            'valid_area': (0, 0, self.target_edge, self.target_edge),
            'hflip': hflip,
            'width_height': (w, h),
        }
        return image, anns, meta


class SquareMix(object):
    def __init__(self, crop, rescale, crop_fraction=0.9):
        self.crop = crop
        self.rescale = rescale
        self.crop_fraction = crop_fraction

    def __call__(self, image, anns):
        if torch.randint(0, 100, (1,)).item() < self.crop_fraction * 100:
            return self.crop(image, anns)

        return self.rescale(image, anns)


class PreserveInput(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, *args):
        return (*args, self.transform(*args))


class NoTransform(object):
    def __call__(self, *args):
        return args
