import argparse
import numpy as np
import torch
import torchvision

import openpifpaf


class Cifar10(openpifpaf.datasets.DataModule):
    root_dir = 'data-cifar10/'
    download = False

    debug = False
    pin_memory = False

    def __init__(self):
        super().__init__()

        categories = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
                      'horse', 'ship', 'truck')
        self.head_metas = [openpifpaf.headmeta.CifDet('cifdet', 'cifar10', categories)]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Cifar10')

        group.add_argument('--cifar10-root-dir',
                           default=cls.root_dir)
        assert not cls.download
        group.add_argument('--cifar10-download',
                           default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocokp specific
        cls.root_dir = args.cifar10_root_dir
        cls.download = args.cifar10_download

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
        return openpifpaf.transforms.Compose([
            self._convert_data,
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
            openpifpaf.transforms.Encoders([enc]),
        ])

    def download_data(self):
        torchvision.datasets.CIFAR10(self.root_dir, download=True)

    def train_loader(self):
        train_data = openpifpaf.datasets.TorchDataset(
            torchvision.datasets.CIFAR10(self.root_dir, train=True, download=self.download),
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = openpifpaf.datasets.TorchDataset(
            torchvision.datasets.CIFAR10(self.root_dir, train=False, download=self.download),
            preprocess=self._preprocess(),
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def eval_loader(self):
        val_data = openpifpaf.datasets.TorchDataset(
            torchvision.datasets.CIFAR10(self.root_dir, train=False, download=self.download),
            preprocess=openpifpaf.transforms.Compose([
                self._convert_data,
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.CenterPadTight(16),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.ToAnnotations([
                    openpifpaf.transforms.ToDetAnnotations(
                        self.head_metas[0].categories),
                ]),
            ]),
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Classification(
            categories=self.head_metas[0].categories,
        )]
