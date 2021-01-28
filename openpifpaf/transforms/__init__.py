"""Transform input data."""

import torchvision

from .annotations import AnnotationJitter, NormalizeAnnotations
from .compose import Compose
from .crop import Crop
from .hflip import HFlip
from .image import Blur, ImageTransform, JpegCompression
from .minsize import MinSize
from .multi_scale import MultiScale
from .pad import CenterPad, CenterPadTight, SquarePad
from .preprocess import Preprocess
from .random import DeterministicEqualChoice, RandomApply
from .rotate import RotateBy90
from .scale import RescaleAbsolute, RescaleRelative, ScaleMix
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
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)),
    RandomApply(JpegCompression(), 0.1),  # maybe irrelevant for COCO, but good for others
    # RandomApply(Blur(), 0.01),  # maybe irrelevant for COCO, but good for others
    ImageTransform(torchvision.transforms.RandomGrayscale(p=0.01)),
    EVAL_TRANSFORM,
])
