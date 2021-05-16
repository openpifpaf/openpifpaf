import argparse
import logging

import PIL
import torch

from . import datasets, decoder, logger, network, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)


class Predictor:
    batch_size = 1
    device_ = None
    fast_rescaling = True
    loader_workers = 1
    long_edge = None

    def __init__(self, checkpoint=None, *, json_data=False, load_image_into_visualizer=False):
        if checkpoint is not None:
            network.Factory.checkpoint = checkpoint
        self.json_data = json_data
        self.load_image_into_visualizer = load_image_into_visualizer

        model_cpu, _ = network.Factory().factory()
        self.model = model_cpu.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            self.model.base_net = model_cpu.base_net
            self.model.head_nets = model_cpu.head_nets

        self.preprocess = self._preprocess_factory()
        self.processor = decoder.factory(model_cpu.head_metas)

        LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
                 self.device, torch.cuda.is_available(), torch.cuda.device_count())

    @property
    def device(self):
        if self.device_ is None:
            if torch.cuda.is_available():
                self.device_ = torch.device('cuda')
            else:
                self.device_ = torch.device('cpu')

        return self.device_

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Predictor')
        group.add_argument('--batch-size', default=cls.batch_size, type=int,
                           help='processing batch size')
        group.add_argument('--long-edge', default=cls.long_edge, type=int,
                           help='rescale the long side of the image (aspect ratio maintained)')
        group.add_argument('--loader-workers', default=None, type=int,
                           help='number of workers for data loading')
        group.add_argument('--precise-rescaling', dest='fast_rescaling',
                           default=True, action='store_false',
                           help='use more exact image rescaling (requires scipy)')
        group.add_argument('--disable-cuda', action='store_true',
                           help='disable CUDA')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.batch_size = args.batch_size
        cls.long_edge = args.long_edge
        cls.loader_workers = (args.loader_workers
                              if args.loader_workers is not None
                              else args.batch_size)
        cls.fast_rescaling = args.fast_rescaling

        if args.disable_cuda:
            cls.device_ = torch.device('cpu')

    def _preprocess_factory(self):
        rescale_t = None
        if self.long_edge:
            rescale_t = transforms.RescaleAbsolute(self.long_edge, fast=self.fast_rescaling)

        pad_t = None
        if self.batch_size > 1:
            assert self.long_edge, '--long-edge must be provided for batch size > 1'
            pad_t = transforms.CenterPad(self.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    def _dataset(self, data):
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.device.type != 'cpu',
            num_workers=self.loader_workers if len(data) > 1 else 0,
            collate_fn=datasets.collate_images_anns_meta)

        for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
            pred_batch = self.processor.batch(self.model, image_tensors_batch, device=self.device)

            # un-batch
            for pred, meta in zip(pred_batch, meta_batch):
                LOG.info('batch %d: %s', batch_i, meta.get('file_name', 'no-file-name'))
                pred = [ann.inverse_transform(meta) for ann in pred]

                # load the original image if necessary
                cpu_image = None
                if self.load_image_into_visualizer:
                    with open(meta['file_name'], 'rb') as f:
                        cpu_image = PIL.Image.open(f).convert('RGB')
                visualizer.Base.image(cpu_image)

                if self.json_data:
                    pred = [ann.json_data() for ann in pred]

                yield pred, meta

    def images(self, file_names):
        data = datasets.ImageList(file_names, preprocess=self.preprocess)
        yield from self._dataset(data)

    def pil_images(self, pil_images):
        data = datasets.PilImageList(pil_images, preprocess=self.preprocess)
        yield from self._dataset(data)

    def numpy_images(self, numpy_images):
        data = datasets.NumpyImageList(numpy_images, preprocess=self.preprocess)
        yield from self._dataset(data)

    def image(self, file_name):
        return next(iter(self.images([file_name])))

    def pil_image(self, image):
        return next(iter(self.pil_images([image])))

    def numpy_image(self, image):
        return next(iter(self.numpy_images([image])))

    def image_file(self, file_pointer):
        pil_image = PIL.Image.open(file_pointer).convert('RGB')
        return self.pil_image(pil_image)
