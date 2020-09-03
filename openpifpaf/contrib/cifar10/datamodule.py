import argparse
import numpy as np
import torch
import torchvision

import openpifpaf


class Cifar10(openpifpaf.datasets.DataModule):
    root_dir = 'data-cifar10/'
    download = False

    n_images = None
    augmentation = True

    def __init__(self):
        super().__init__()

        categories = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                      'horse', 'ship', 'truck')
        self.head_metas = (openpifpaf.headmeta.CifDet('cifdet', 'cifar10', categories),)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
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
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocokp specific
        cls.root_dir = args.cifar10_root_dir
        cls.download = args.cifar10_download

        cls.n_images = args.cifar10_n_images
        cls.augmentation = args.cifar10_augmentation

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([5, 5, 21, 21], dtype=np.float32),
            'category_id': category_id + 1,
        }]

        return image, anns, meta

    def _preprocess(self):
        enc = openpifpaf.encoder.CifDet(self.head_metas[0])

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                self._convert_data,
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(33),
                openpifpaf.transforms.ImageTransform(torchvision.transforms.ToTensor()),
                openpifpaf.transforms.ImageTransform(
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[0.5, 0.5, 0.5]),
                ),
                openpifpaf.transforms.Encoders([enc]),
            ])

        return openpifpaf.transforms.Compose([
            self._convert_data,
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RescaleAbsolute(33),
            openpifpaf.transforms.ImageTransform(torchvision.transforms.ToTensor()),
            openpifpaf.transforms.ImageTransform(
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5]),
            ),
            openpifpaf.transforms.Encoders([enc]),
        ])

    def train_loader(self):
        train_data = openpifpaf.datasets.TorchDataset(
            torchvision.datasets.CIFAR10(self.root_dir, train=True, download=self.download),
            preprocess=self._preprocess(),
        )
        if self.n_images:
            train_data = torch.utils.data.Subset(train_data, indices=range(self.n_images))
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = openpifpaf.datasets.TorchDataset(
            torchvision.datasets.CIFAR10( self.root_dir, train=False, download=self.download),
            preprocess=self._preprocess(),
        )
        if self.n_images:
            val_data = torch.utils.data.Subset(val_data, indices=range(self.n_images))
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)
