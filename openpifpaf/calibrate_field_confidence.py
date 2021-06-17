"""Calibrate."""

import argparse
from collections import defaultdict
import logging
import time

import numpy as np
import torch

from . import datasets, decoder, logger, network, show, visualizer, __version__

LOG = logging.getLogger(__name__)


def default_output_name(args):
    output = '{}.calibrated-{}'.format(network.Factory.checkpoint, args.dataset)

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

    return output + '.pkl'


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.calibrate_field_confidence',
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
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--n-batches', default=None, type=int,
                        help='restrict number of batches')
    args = parser.parse_args()

    logger.configure(args, LOG)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False

    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def calibrate(datamodule, model, *, n_batches=None):
    total_start = time.perf_counter()
    loop_start = time.perf_counter()

    histograms = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: torch.zeros((100,))
            )
        )
    )

    loader = datamodule.train_loader()
    for batch_i, (processed_image_batch, gt_batch, _) in enumerate(loader):
        if n_batches and batch_i > n_batches:
            break

        LOG.info('batch %d / %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, len(loader), time.perf_counter() - loop_start,
                 batch_i / max(1, (time.perf_counter() - total_start)))
        loop_start = time.perf_counter()

        with torch.no_grad():
            pred_batch = model(processed_image_batch)

        for cf_i, (pred_field, gt_field) in enumerate(zip(pred_batch, gt_batch)):
            for field_i in range(gt_field.shape[1]):
                pred_f = pred_field[:, field_i]
                pred_f_logit = -torch.log((1.0 / (pred_f + 1e-6)) - 1)
                pred_f_logit = torch.clamp(pred_f_logit, -5.0, 5.0)
                gt_f = gt_field[:, field_i]
                for gt_target in (0, 1):
                    mask = (gt_f == gt_target)
                    hist = torch.histc(pred_f_logit[mask], 100, -5.0, 5.0)
                    histograms[cf_i][field_i][gt_target] += hist

    total_time = time.perf_counter() - total_start
    print(f'total time: {total_time}')

    x_bins = np.linspace(-5.0, 5.0, 100)
    with show.canvas(nrows=2, ncols=3) as axes:
        axes[0][0].set_ylabel('CIF bg')
        axes[0][1].set_ylabel('CIF fg')
        axes[0][2].set_ylabel('CIF ratio')
        axes[1][0].set_ylabel('CAF bg')
        axes[1][1].set_ylabel('CAF fg')
        axes[1][2].set_ylabel('CAF ratio')
        for (ax_bg, ax_fg, ax_ratio), cf_h in zip(axes, histograms.values()):
            for field_i, field_h in cf_h.items():
                ax_bg.plot(x_bins, field_h[0])
                ax_fg.plot(x_bins, field_h[1])
                ax_ratio.plot(x_bins, (field_h[1] + 0.1) / (field_h[0] + field_h[1] + 0.1))

        # rescale bg graphs
        axes[0][0].set_ylim(*axes[0][1].get_ylim())
        axes[1][0].set_ylim(*axes[1][1].get_ylim())

    corrections = []
    for cf_i, cf_h in histograms.items():
        n_fields = len(cf_h)
        correction = torch.nn.Conv2d(n_fields, n_fields, 1, groups=n_fields)
        corrections.append(correction)
        for field_i, field_h in cf_h.items():
            if field_h[1][1:].sum() > 0:
                hist_fg = field_h[1].numpy()
                hist_fg /= np.sum(hist_fg)
                mean = np.sum(x_bins[1:] * hist_fg[1:])
                std_dev = np.sqrt(np.sum((x_bins[1:] - mean)**2 * hist_fg[1:]))
                print(f'{cf_i}, {field_i}: {mean} +/- {std_dev}')

                # 1.3 just seems a natural value looking at uncorrected values
                weight = np.clip(1.3 / std_dev, 0.8, 1.25)
                bias = np.clip(1.0 - mean * weight, -1.0, 1.0)
            else:
                weight = 1.0
                bias = 0.0

            correction.weight.data[field_i, 0, 0, 0] = weight
            correction.bias.data[field_i] = bias
        print(f'{cf_i}: weight = {correction.weight.data[:, 0, 0, 0]}, bias = {correction.bias.data}')

    return corrections


def main():
    args = cli()

    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    datamodule = datasets.factory(args.dataset)
    model, epoch = network.Factory().factory(head_metas=datamodule.head_metas)

    # determine calibrations
    corrections = calibrate(datamodule, model, n_batches=args.n_batches)

    # set corrections
    for head_net, correction in zip(model.head_nets, corrections):
        head_net.confidence_calibration = correction

    # write output checkpoint
    print('output:', args.output)
    torch.save({
        'model': model,
        'epoch': epoch,
        'meta': None,
    }, args.output)

    # to check calibration, run again
    calibrate(datamodule, model, n_batches=args.n_batches)


if __name__ == '__main__':
    main()
