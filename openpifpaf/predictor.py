import argparse
import logging

import PIL
import torch

from . import datasets, decoder, network, transforms, visualizer

LOG = logging.getLogger(__name__)


class Predictor:
    device_ = None
    fast_rescaling = True
    long_edge = None

    def __init__(self, checkpoint=None, head_metas=None, *,
                 json_data=False, load_image_into_visualizer=False):
        if checkpoint is not None:
            network.Factory.checkpoint = checkpoint
        self.json_data = json_data
        self.load_image_into_visualizer = load_image_into_visualizer

        self.model_cpu, _ = network.Factory().factory(head_metas=head_metas)
        self.model = self.model_cpu.to(self.device)
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)
            self.model.base_net = self.model_cpu.base_net
            self.model.head_nets = self.model_cpu.head_nets

        self.preprocess = self._preprocess_factory()
        self.processor = decoder.factory(self.model_cpu.head_metas)

        self.last_decoder_time = 0.0
        self.last_nn_time = 0.0
        self.total_nn_time = 0.0
        self.total_decoder_time = 0.0
        self.total_images = 0

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
        group.add_argument('--long-edge', default=cls.long_edge, type=int,
                           help='rescale the long side of the image (aspect ratio maintained)')
        group.add_argument('--precise-rescaling', dest='fast_rescaling',
                           default=True, action='store_false',
                           help='use more exact image rescaling (requires scipy)')
        group.add_argument('--disable-cuda', action='store_true',
                           help='disable CUDA')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.long_edge = args.long_edge
        cls.fast_rescaling = args.fast_rescaling

        if args.disable_cuda:
            cls.device_ = torch.device('cpu')

    def _preprocess_factory(self):
        rescale_t = None
        if self.long_edge:
            rescale_t = transforms.RescaleAbsolute(self.long_edge, fast=self.fast_rescaling)

        pad_t = None
        if self.long_edge is not None:
            pad_t = transforms.CenterPad(self.long_edge)
        else:
            pad_t = transforms.CenterPadTight(16)

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
        ])

    def dataset(self, data, *, batch_size=1, loader_workers=0):
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False,
            pin_memory=self.device.type != 'cpu',
            num_workers=loader_workers,
            collate_fn=datasets.collate_images_anns_meta)

        yield from self.dataloader(dataloader)

    def dataloader(self, dataloader):
        for batch_i, (image_tensors_batch, gt_anns_batch, meta_batch) in enumerate(dataloader):
            pred_batch = self.processor.batch(self.model, image_tensors_batch, device=self.device)
            self.last_decoder_time = self.processor.last_decoder_time
            self.last_nn_time = self.processor.last_nn_time
            self.total_decoder_time += self.processor.last_decoder_time
            self.total_nn_time += self.processor.last_nn_time
            self.total_images += len(image_tensors_batch)

            # un-batch
            for pred, gt_anns, meta in zip(pred_batch, gt_anns_batch, meta_batch):
                LOG.info('batch %d: %s', batch_i, meta.get('file_name', 'no-file-name'))
                pred = [ann.inverse_transform(meta) for ann in pred]
                gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]

                # load the original image if necessary
                cpu_image = None
                if self.load_image_into_visualizer:
                    with open(meta['file_name'], 'rb') as f:
                        cpu_image = PIL.Image.open(f).convert('RGB')
                visualizer.Base.image(cpu_image)

                if self.json_data:
                    pred = [ann.json_data() for ann in pred]

                yield pred, gt_anns, meta

    def images(self, file_names, **kwargs):
        data = datasets.ImageList(file_names, preprocess=self.preprocess)
        yield from self.dataset(data, **kwargs)

    def pil_images(self, pil_images, **kwargs):
        data = datasets.PilImageList(pil_images, preprocess=self.preprocess)
        yield from self.dataset(data, **kwargs)

    def numpy_images(self, numpy_images, **kwargs):
        data = datasets.NumpyImageList(numpy_images, preprocess=self.preprocess)
        yield from self.dataset(data, **kwargs)

    def image(self, file_name):
        return next(iter(self.images([file_name])))

    def pil_image(self, image):
        return next(iter(self.pil_images([image])))

    def numpy_image(self, image):
        return next(iter(self.numpy_images([image])))

    def image_file(self, file_pointer):
        pil_image = PIL.Image.open(file_pointer).convert('RGB')
        return self.pil_image(pil_image)
