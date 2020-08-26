"""Datasets and tools to load data in batches."""

from .cifar10 import Cifar10
from .coco import Coco
from .cocodet import CocoDet
from .cocokp import CocoKp
from .collate import collate_images_anns_meta, collate_images_targets_meta
from .factory import cli, configure, factory, DATAMODULES
from .image_list import ImageList, PilImageList
