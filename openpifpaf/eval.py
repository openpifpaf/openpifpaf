"""Evaluation on COCO data."""

import argparse
import json
import logging
import os
import sys
import time

import PIL
import thop
import torch

from . import datasets, decoder, network, plugins, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)


def default_output_name(args):
    output = '{}.eval-{}'.format(args.checkpoint, args.dataset)

    # coco
    if args.coco_eval_orientation_invariant or args.coco_eval_extended_scale:
        output += '-coco'
        if args.coco_eval_orientation_invariant:
            output += 'o'
        if args.coco_eval_extended_scale:
            output += 's'
    if args.coco_eval_long_edge is not None and args.coco_eval_long_edge != 641:
        output += '-cocoedge{}'.format(args.coco_eval_long_edge)

    if args.two_scale:
        output += '-twoscale'
    if args.multi_scale:
        output += '-multiscale'
        if args.multi_scale_hflip:
            output += 'whflip'

    return output


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    plugins.register()
    datasets.cli(parser)
    decoder.cli(parser)
    network.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--no-skip-epoch0', dest='skip_epoch0',
                        default=True, action='store_false',
                        help='do not skip eval for epoch 0')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--show-final-image', default=False, action='store_true')

    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')

    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO if not args.debug else logging.DEBUG
    if args.quiet:
        assert not args.debug
        log_level = logging.WARNING
    if args.log_stats:
        # pylint: disable=import-outside-toplevel
        from pythonjsonlogger import jsonlogger
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            jsonlogger.JsonFormatter('(message) (levelname) (name)'))
        logging.basicConfig(handlers=[stdout_handler])
        logging.getLogger('openpifpaf').setLevel(log_level)
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)
        LOG.setLevel(log_level)
    else:
        logging.basicConfig()
        logging.getLogger('openpifpaf').setLevel(log_level)
        LOG.setLevel(log_level)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    datasets.configure(args)
    decoder.configure(args)
    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def count_ops(model, height=641, width=641):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, height, width, device=device)
    gmacs, params = thop.profile(model, inputs=(dummy_input, ))
    LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
    return gmacs, params


# pylint: disable=too-many-statements
def main():
    args = cli()

    # skip existing?
    if args.skip_epoch0:
        if args.checkpoint.endswith('.epoch000'):
            print('Not evaluating epoch 0.')
            return
    if args.skip_existing:
        stats_file = args.output + '.stats.json'
        if os.path.exists(stats_file):
            print('Output file {} exists already. Exiting.'.format(stats_file))
            return
        print('{} not found. Processing: {}'.format(stats_file, args.checkpoint))

    datamodule = datasets.factory(args.dataset)
    model_cpu, _ = network.factory_from_args(args, head_metas=datamodule.head_metas)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(
        head_metas, profile=args.profile_decoder, profile_device=args.device)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    metrics = datamodule.metrics()
    total_start = time.time()
    loop_start = time.time()
    nn_time = 0.0
    decoder_time = 0.0
    n_images = 0

    loader = datamodule.eval_loader()
    for batch_i, (image_tensors, anns_batch, meta_batch) in enumerate(loader):
        LOG.info('batch %d / %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, len(loader), time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        loop_start = time.time()

        pred_batch = processor.batch(model, image_tensors,
                                     device=args.device, gt_anns_batch=anns_batch)
        n_images += len(image_tensors)
        decoder_time += processor.last_decoder_time
        nn_time += processor.last_nn_time

        # loop over batch
        assert len(image_tensors) == len(meta_batch)
        for pred, gt_anns, image_meta in zip(pred_batch, anns_batch, meta_batch):
            pred = transforms.Preprocess.annotations_inverse(pred, image_meta)
            for metric in metrics:
                metric.accumulate(pred, image_meta, ground_truth=gt_anns)

            if args.show_final_image:
                # show ground truth and predictions on original image
                gt_anns = transforms.Preprocess.annotations_inverse(gt_anns, image_meta)

                annotation_painter = show.AnnotationPainter()
                with open(image_meta['local_file_path'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

                with show.image_canvas(cpu_image) as ax:
                    annotation_painter.annotations(ax, gt_anns, color='grey')
                    annotation_painter.annotations(ax, pred)

    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')

    # model stats
    counted_ops = list(count_ops(model_cpu))
    local_checkpoint = network.local_checkpoint_path(args.checkpoint)
    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write
    for metric in metrics:
        if args.write_predictions:
            metric.write_predictions(args.output)

        additional_data = {
            'dataset': args.dataset,
            'total_time': total_time,
            'checkpoint': args.checkpoint,
            'count_ops': counted_ops,
            'file_size': file_size,
            'n_images': n_images,
            'decoder_time': decoder_time,
            'nn_time': nn_time,
        }
        stats = dict(**metric.stats(), **additional_data)
        with open(args.output + '.stats.json', 'w') as f:
            json.dump(stats, f)

        LOG.info('stats:\n%s', json.dumps(stats, indent=4))
        LOG.info(
            'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
            1000 * stats['decoder_time'] / stats['n_images'],
            1000 * stats['nn_time'] / stats['n_images'],
            1000 * stats['total_time'] / stats['n_images'],
        )


if __name__ == '__main__':
    main()
