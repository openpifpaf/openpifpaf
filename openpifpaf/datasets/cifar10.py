import numpy as np
import torch
import torchvision

from .module import DataModule
from ..network import headmeta
from .. import transforms
from .collate import collate_images_targets_meta
from .torch_dataset import TorchDataset


class Cifar10(DataModule):
    description = 'Cifar10 data module.'

    # cli configurable
    root_dir = 'data-cifar10/'
    download = False

    n_images = None
    augmentation = True

    @classmethod
    def cli(cls, parser):
        group = parser.add_argument_group('data module Cifar10')

        group.add_argument('--cifar10-root-dir',
                           default=cls.root_dir)
        assert not cls.download
        group.add_argument('--cifar10-download',
                           default=False, action='store_true')

        group.add_argument('--cifar10-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        assert cls.augmentation
        group.add_argument('--cifar10-no-augmentation',
                           dest='cifar10_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')

    @classmethod
    def configure(cls, args):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocokp specific
        cls.root_dir = args.cifar10_root_dir
        cls.download = args.cifar10_download

        cls.n_images = args.cifar10_n_images
        cls.augmentation = args.cifar10_augmentation

    def head_metas(self):
        categories = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                      'horse', 'ship', 'truck')
        return (headmeta.Detection('cifdet', categories),)

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([0, 0, 31, 31], dtype=np.float32),
            'category_id': category_id,
        }]

        return image, anns, meta

    @classmethod
    def _preprocess(cls):
        if not cls.augmentation:
            return transforms.Compose([
                cls._convert_data,
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(33),
                transforms.EVAL_TRANSFORM,
            ])

        return transforms.Compose([
            cls._convert_data,
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(33),
            transforms.TRAIN_TRANSFORM,
        ])

    def train_loader(self, target_transforms):
        train_data = TorchDataset(
            torchvision.datasets.CIFAR10(self.root_dir, train=True, download=self.download),
            preprocess=self._preprocess(),
            target_transforms=target_transforms,
        )
        if self.n_images:
            train_data = torch.utils.data.Subset(train_data, indices=range(self.n_images))
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)

    def val_loader(self, target_transforms):
        val_data = torchvision.datasets.CIFAR10(
            self.root_dir,
            train=False,
            transform=self._preprocess(),
            target_transform=target_transforms,
            download=self.download,
        )
        if self.n_images:
            val_data = torch.utils.data.Subset(val_data, indices=range(self.n_images))
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta)
