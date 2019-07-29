"""Benchmark."""

import argparse
import datetime
import json
import logging
import os
import subprocess

import pysparkling


DEFAULT_BACKBONES = [
    'shufflenetv2x1',
    'shufflenetv2x2',
    'resnet50',
    'resnext50',
    'resnet101',
    'resnet152',
]


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--output', default=None,
                        help='output file name')
    parser.add_argument('--backbones', default=DEFAULT_BACKBONES, nargs='+',
                        help='backbones to evaluate')
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args, eval_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # default eval_args
    if not eval_args:
        eval_args = ['--all-images', '--loader-workers=8']

    # generate a default output filename
    if args.output is None:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = 'outputs/benchmark-{}/'.format(now)
        os.makedirs(args.output)

    return args, eval_args


def run_eval_coco(output_folder, backbone, eval_args):
    out_file = os.path.join(output_folder, backbone)
    if os.path.exists(out_file + '.stats.json'):
        logging.warning('Output file %s exists already. Skipping.',
                        out_file + '.stats.json')
        return

    logging.debug('Launching eval for %s.', backbone)
    subprocess.run([
        'python', '-m', 'openpifpaf.eval_coco',
        '--output', out_file,
        '--checkpoint', backbone,
    ] + eval_args)


def main():
    args, eval_args = cli()

    for backbone in args.backbones:
        run_eval_coco(args.output, backbone, eval_args)

    sc = pysparkling.Context()
    stats = (
        sc
        .textFile(args.output + '*.stats.json')
        .map(json.loads)
        .map(lambda d: (d['checkpoint'], d))
        .collectAsMap()
    )
    logging.debug('all data: %s', stats)

    # pretty printing
    for backbone, data in sorted(stats.items(), key=lambda b_d: b_d[1]['stats'][0]):
        print(
            '| {backbone: <15} '
            '| __{AP:.1f}__ '
            '| {APM: <8.1f} '
            '| {APL: <8.1f} '
            '| {t: <15.0f} '
            '| {tdec: <12.0f} |'
            ''.format(
                backbone=backbone,
                AP=100.0 * data['stats'][0],
                APM=100.0 * data['stats'][3],
                APL=100.0 * data['stats'][4],
                t=1000.0 * data['total_time'] / data['n_images'],
                tdec=1000.0 * data['decoder_time'] / data['n_images'],
            )
        )


if __name__ == '__main__':
    main()
