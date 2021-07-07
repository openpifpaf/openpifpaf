"""Transform input data."""

import torchvision

from . import pair
from .annotations import AnnotationJitter, NormalizeAnnotations
from .assertion import Assert
from .compose import Compose
from .crop import Crop
from .deinterlace import Deinterlace
from .encoders import Encoders
from .hflip import HFlip
from .image import Blur, HorizontalBlur, ImageTransform, JpegCompression
from .impute import AddCrowdForIncompleteHead
from .minsize import MinSize
from .multi_scale import MultiScale
from .pad import CenterPad, CenterPadTight, SquarePad
from .preprocess import Preprocess
from .random import DeterministicEqualChoice, RandomApply, RandomChoice
from .rotate import RotateBy90, RotateUniform
from .scale import RescaleAbsolute, RescaleRelative, ScaleMix
from .toannotations import ToAnnotations, ToCrowdAnnotations, ToDetAnnotations, ToKpAnnotations
from .unclipped import UnclippedArea, UnclippedSides


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
        brightness=0.4, contrast=0.1, saturation=0.4, hue=0.1)),
    RandomApply(JpegCompression(), 0.1),  # maybe irrelevant for COCO, but good for others
    # RandomApply(Blur(), 0.01),  # maybe irrelevant for COCO, but good for others
    ImageTransform(torchvision.transforms.RandomGrayscale(p=0.01)),
    EVAL_TRANSFORM,
])


__all__ = [
    'Preprocess',
    'AnnotationJitter', 'NormalizeAnnotations',
    'Assert',
    'Compose',
    'Crop',
    'Deinterlace',
    'Encoders',
    'HFlip',
    'Blur', 'HorizontalBlur', 'ImageTransform', 'JpegCompression',
    'AddCrowdForIncompleteHead',
    'MinSize',
    'MultiScale',
    'CenterPad', 'CenterPadTight', 'SquarePad',
    'DeterministicEqualChoice', 'RandomApply', 'RandomChoice',
    'RotateBy90', 'RotateUniform',
    'RescaleAbsolute', 'RescaleRelative', 'ScaleMix',
    'ToAnnotations', 'ToCrowdAnnotations', 'ToDetAnnotations', 'ToKpAnnotations',
    'UnclippedArea', 'UnclippedSides',
]
