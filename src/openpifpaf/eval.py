"""Evaluation on COCO data."""

import argparse
from collections import defaultdict
import glob
import json
import logging
import os
import sys
import time
import typing as t

import PIL.Image
import torch

from . import datasets, decoder, logger, network, show, visualizer, __version__
from .configurable import Configurable
from .predictor import Predictor

try:
    import thop
except ImportError:
    thop = None

LOG = logging.getLogger(__name__)


def count_ops(model, height=641, width=641):
    if thop is None:
        print('warning: run "pip3 install thop" to count parameters and ops')
        return -1, -1
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, height, width, device=device)
    gmacs, params = thop.profile(model, inputs=(dummy_input, ))  # pylint: disable=unbalanced-tuple-unpacking
    LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
    return gmacs, params


class Evaluator(Configurable):
    args: t.Optional[argparse.Namespace] = None
    loader_warmup = 3.0
    n_images: t.Optional[int] = None
    show_final_image = False
    show_final_ground_truth = False
    skip_epoch0 = True
    skip_existing = True
    write_predictions = False

    def __init__(self, dataset_name: str, **kwargs):
        super().__init__(**kwargs)

        self.dataset_name = dataset_name
        self.datamodule = datasets.factory(dataset_name)
        self.data_loader = self.datamodule.eval_loader()

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Evaluator')
        group.add_argument('--eval-loader-warmup', default=cls.loader_warmup, type=float)
        group.add_argument('--eval-n-images', default=cls.n_images, type=int)
        assert not cls.show_final_image
        group.add_argument('--eval-show-final-image', default=False, action='store_true',
                           help='show the final image')
        assert not cls.show_final_ground_truth
        group.add_argument('--eval-show-final-ground-truth', default=False, action='store_true',
                           help='show the final image with ground truth annotations')
        assert cls.skip_epoch0
        group.add_argument('--eval-no-skip-epoch0', dest='eval_skip_epoch0',
                           default=True, action='store_false',
                           help='do not skip eval for epoch 0')
        assert cls.skip_existing
        group.add_argument('--eval-no-skip-existing', dest='eval_skip_existing',
                           default=True, action='store_false',
                           help='skip if output eval file exists already')
        assert not cls.write_predictions
        group.add_argument('--eval-write-predictions', default=False, action='store_true',
                           help='write a json and a zip file of the predictions')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.args = args
        cls.loader_warmup = args.eval_loader_warmup
        cls.n_images = args.eval_n_images
        cls.show_final_image = args.eval_show_final_image
        cls.show_final_ground_truth = args.eval_show_final_ground_truth
        cls.skip_epoch0 = args.eval_skip_epoch0
        cls.skip_existing = args.eval_skip_existing
        cls.write_predictions = args.eval_write_predictions

    def default_output_name(self, args: argparse.Namespace) -> str:
        output = '{}.eval-{}'.format(network.Factory.checkpoint, self.dataset_name)

        # coco
        if args.coco_eval_orientation_invariant or args.coco_eval_extended_scale:
            output += '-coco'
            if args.coco_eval_orientation_invariant:
                output += 'o'
            if args.coco_eval_extended_scale:
                output += 's'
        if args.coco_eval_long_edge is not None and args.coco_eval_long_edge != 641:
            output += '-cocoedge{}'.format(args.coco_eval_long_edge)

        # dense
        if args.dense_connections:
            output += '-dense'
            if args.dense_connections != 1.0:
                output += '{}'.format(args.dense_connections)

        return output

    def accumulate(self, predictor, metrics):
        prediction_loader = predictor.enumerated_dataloader(enumerate(self.data_loader))
        if self.loader_warmup:
            LOG.info('Data loader warmup (%.1fs) ...', self.loader_warmup)
            time.sleep(self.loader_warmup)
            LOG.info('Done.')

        total_start = time.perf_counter()
        loop_start = time.perf_counter()

        for image_i, (pred, gt_anns, image_meta) in enumerate(prediction_loader):
            LOG.info('image %d / %d, last loop: %.3fs, images per second=%.1f',
                     image_i, len(self.data_loader), time.perf_counter() - loop_start,
                     image_i / max(1, (time.perf_counter() - total_start)))
            loop_start = time.perf_counter()

            for metric in metrics:
                metric.accumulate(pred, image_meta, ground_truth=gt_anns)

            if self.show_final_image:
                # show ground truth and predictions on original image
                annotation_painter = show.AnnotationPainter()
                with open(image_meta['local_file_path'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

                with show.image_canvas(cpu_image) as ax:
                    if self.show_final_ground_truth:
                        annotation_painter.annotations(ax, gt_anns, color='grey')
                    annotation_painter.annotations(ax, pred)

            if self.n_images is not None and image_i >= self.n_images - 1:
                break

        total_time = time.perf_counter() - total_start
        return total_time

    def evaluate(self, output: t.Optional[str]):
        # generate a default output filename
        if output is None:
            assert self.args is not None
            output = self.default_output_name(self.args)

        # skip existing?
        if self.skip_epoch0:
            assert network.Factory.checkpoint is not None
            if network.Factory.checkpoint.endswith('.epoch000'):
                print('Not evaluating epoch 0.')
                return
        if self.skip_existing:
            stats_file = output + '.stats.json'
            if os.path.exists(stats_file):
                print('Output file {} exists already. Exiting.'.format(stats_file))
                return
            print('{} not found. Processing: {}'.format(stats_file, network.Factory.checkpoint))

        predictor = Predictor(head_metas=self.datamodule.head_metas)
        metrics = self.datamodule.metrics()

        total_time = self.accumulate(predictor, metrics)

        # model stats
        counted_ops = list(count_ops(predictor.model_cpu))
        local_checkpoint = network.local_checkpoint_path(network.Factory.checkpoint)
        file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

        # write
        additional_data = {
            'args': sys.argv,
            'version': __version__,
            'dataset': self.dataset_name,
            'total_time': total_time,
            'checkpoint': network.Factory.checkpoint,
            'count_ops': counted_ops,
            'file_size': file_size,
            'n_images': predictor.total_images,
            'decoder_time': predictor.total_decoder_time,
            'nn_time': predictor.total_nn_time,
        }

        metric_stats = defaultdict(list)
        for metric in metrics:
            if self.write_predictions:
                metric.write_predictions(output, additional_data=additional_data)

            this_metric_stats = metric.stats()
            assert (len(this_metric_stats.get('text_labels', []))
                    == len(this_metric_stats.get('stats', [])))

            for k, v in this_metric_stats.items():
                metric_stats[k] = metric_stats[k] + v

        stats = dict(**metric_stats, **additional_data)

        # write stats file
        with open(output + '.stats.json', 'w') as f:
            json.dump(stats, f)

        LOG.info('stats:\n%s', json.dumps(stats, indent=4))
        LOG.info(
            'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
            1000 * stats['decoder_time'] / stats['n_images'],
            1000 * stats['nn_time'] / stats['n_images'],
            1000 * stats['total_time'] / stats['n_images'],
        )

    def watch(self, checkpoint_pattern: str, timeout: float):
        evaluated_pattern = '{}*eval-{}.stats.json'.format(checkpoint_pattern, self.dataset_name)

        while True:
            # find checkpoints that have not been evaluated
            last_check = time.time()
            all_checkpoints = glob.glob(checkpoint_pattern)
            evaluated = glob.glob(evaluated_pattern)
            if self.skip_epoch0:
                all_checkpoints = [c for c in all_checkpoints
                                   if not c.endswith('.epoch000')]
            checkpoints = [c for c in all_checkpoints
                           if not any(e.startswith(c) for e in evaluated)]
            LOG.info('%d checkpoints, %d evaluated, %d todo: %s',
                     len(all_checkpoints), len(evaluated), len(checkpoints), checkpoints)

            # evaluate all checkpoints
            for checkpoint in checkpoints:
                # reset
                network.Factory.checkpoint = checkpoint

                self.evaluate(None)

            # wait before looking for more work
            time.sleep(max(1.0, timeout - (time.time() - last_check)))


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    datasets.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser, skip_batch_size=True, skip_loader_workers=True)
    show.cli(parser)
    visualizer.cli(parser)
    Evaluator.cli(parser)

    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--watch', default=False, const=60, nargs='?', type=int,
                        help=('Watch a directory for new checkpoint files. '
                              'Optionally specify the number of seconds between checks.')
                        )
    args = parser.parse_args()

    logger.configure(args, LOG)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)
    Evaluator.configure(args)

    return args


def main():
    args = cli()
    evaluator = Evaluator(args.dataset)

    if args.watch:
        assert args.output is None
        evaluator.watch(args.checkpoint, args.watch)
    else:
        evaluator.evaluate(args.output)


if __name__ == '__main__':
    main()
