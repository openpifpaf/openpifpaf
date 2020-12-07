"""Evaluation on COCO data."""

import argparse
import glob
import json
import logging
import os
import sys
import time

import PIL
import thop
import torch

from . import datasets, decoder, logger, network, plugin, show, transforms, visualizer, __version__

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

    # dense
    if args.dense_connections:
        output += '-dense'
        if args.dense_connections != 1.0:
            output += '{}'.format(args.dense_connections)

    if args.two_scale:
        output += '-twoscale'

    return output


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    plugin.register()

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
    parser.add_argument('--watch', default=False, const=60, nargs='?', type=int)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--show-final-image', default=False, action='store_true')
    parser.add_argument('--show-final-ground-truth', default=False, action='store_true')
    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

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


# pylint: disable=too-many-statements,too-many-branches
def evaluate(args):
    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

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
    processor = decoder.factory(head_metas)
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
                    if args.show_final_ground_truth:
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
        additional_data = {
            'args': sys.argv,
            'version': __version__,
            'dataset': args.dataset,
            'total_time': total_time,
            'checkpoint': args.checkpoint,
            'count_ops': counted_ops,
            'file_size': file_size,
            'n_images': n_images,
            'decoder_time': decoder_time,
            'nn_time': nn_time,
        }

        if args.write_predictions:
            metric.write_predictions(args.output, additional_data=additional_data)

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


def watch(args):
    assert args.output is None
    pattern = args.checkpoint
    evaluated_pattern = '{}*eval-{}.stats.json'.format(pattern, args.dataset)

    while True:
        # find checkpoints that have not been evaluated
        all_checkpoints = glob.glob(pattern)
        evaluated = glob.glob(evaluated_pattern)
        if args.skip_epoch0:
            all_checkpoints = [c for c in all_checkpoints
                               if not c.endswith('.epoch000')]
        checkpoints = [c for c in all_checkpoints
                       if not any(e.startswith(c) for e in evaluated)]
        LOG.info('%d checkpoints, %d evaluated, %d todo: %s',
                 len(all_checkpoints), len(evaluated), len(checkpoints), checkpoints)

        # evaluate all checkpoints
        for checkpoint in checkpoints:
            # reset
            args.output = None
            args.checkpoint = checkpoint

            evaluate(args)

        # wait before looking for more work
        time.sleep(args.watch)


def main():
    args = cli()

    if args.watch:
        watch(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
