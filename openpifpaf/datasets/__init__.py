"""Datasets and tools to load data in batches."""

from .coco import Coco
from .cocodet import CocoDet
from .cocokp import CocoKp
from .collate import collate_images_anns_meta, collate_images_targets_meta
from .factory import DATAMODULES, datamodule_factory, train_cli, train_configure
from .image_list import ImageList, PilImageList

DATAMODULES.add(CocoDet)
DATAMODULES.add(CocoKp)
