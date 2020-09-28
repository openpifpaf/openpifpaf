"""Benchmark."""

import argparse
import datetime
import json
import logging
import os
import subprocess

import pysparkling

from . import __version__

LOG = logging.getLogger(__name__)


DEFAULT_CHECKPOINTS = [
    'resnet50',
    'shufflenetv2k16',
    'shufflenetv2k30',
]


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.benchmark',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    parser.add_argument('--output', default=None,
                        help='output file name')
    parser.add_argument('--checkpoints', default=DEFAULT_CHECKPOINTS, nargs='+',
                        help='checkpoints to evaluate')
    parser.add_argument('--iccv2019-ablation', default=False, action='store_true')
    parser.add_argument('--v012-ablation', default=False, action='store_true')
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args, eval_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # default eval_args
    if not eval_args:
        eval_args = ['--loader-workers=8']

    # default loader workers
    if not any(l.startswith('--loader-workers') for l in eval_args):
        LOG.info('adding "--loader-workers=8" to the argument list')
        eval_args.append('--loader-workers=8')

    # default dataset
    if not any(l.startswith('--dataset') for l in eval_args):
        LOG.info('adding "--dataset=cocokp" to the argument list')
        eval_args.append('--dataset=cocokp')
        if not any(l.startswith('--coco-no-eval-annotation-filter') for l in eval_args):
            LOG.info('adding "--coco-no-eval-annotation-filter" to the argument list')
            eval_args.append('--coco-no-eval-annotation-filter')
        if not any(l.startswith('--force-complete-pose') for l in eval_args):
            LOG.info('adding "--force-complete-pose" to the argument list')
            eval_args.append('--force-complete-pose')
        if not any(l.startswith('--seed-threshold') for l in eval_args):
            LOG.info('adding "--seed-threshold=0.2" to the argument list')
            eval_args.append('--seed-threshold=0.2')

    # generate a default output filename
    if args.output is None:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = 'outputs/benchmark-{}/'.format(now)
        os.makedirs(args.output)

    return args, eval_args


def run_eval_coco(output_folder, checkpoint, eval_args, output_name=None):
    if output_name is None:
        output_name = checkpoint
    output_name = output_name.replace('/', '-')

    out_file = os.path.join(output_folder, output_name)
    if os.path.exists(out_file + '.stats.json'):
        LOG.warning('Output file %s exists already. Skipping.',
                    out_file + '.stats.json')
        return

    LOG.debug('Launching eval for %s.', output_name)
    subprocess.run([
        'python', '-m', 'openpifpaf.eval',
        '--output', out_file,
        '--checkpoint', checkpoint,
    ] + eval_args, check=True)


def main():
    args, eval_args = cli()

    if args.iccv2019_ablation:
        assert len(args.checkpoints) == 1
        multi_eval_args = [
            eval_args,
            eval_args + ['--connection-method=blend'],
            eval_args + ['--connection-method=blend', '--long-edge=961', '--multi-scale',
                         '--no-multi-scale-hflip'],
            eval_args + ['--connection-method=blend', '--long-edge=961', '--multi-scale'],
        ]
        names = [
            'singlescale-max',
            'singlescale',
            'multiscale-nohflip',
            'multiscale',
        ]
        for eval_args_i, name_i in zip(multi_eval_args, names):
            run_eval_coco(args.output, args.checkpoints[0], eval_args_i, output_name=name_i)
    elif args.v012_ablation:
        multi_eval_args = [
            eval_args,
            eval_args + ['--greedy'],
            eval_args + ['--greedy', '--dense-connections'],
            eval_args + ['--dense-connections'],
            eval_args + ['--dense-connections=0.1'],
        ]
        for checkpoint in args.checkpoints:
            names = [
                checkpoint,
                '{}.greedy'.format(checkpoint),
                '{}.greedy.dense'.format(checkpoint),
                '{}.dense'.format(checkpoint),
                '{}.dense.hierarchy'.format(checkpoint),
            ]
            for eval_args_i, name_i in zip(multi_eval_args, names):
                run_eval_coco(args.output, checkpoint, eval_args_i, output_name=name_i)
    else:
        for checkpoint in args.checkpoints:
            run_eval_coco(args.output, checkpoint, eval_args)

    sc = pysparkling.Context()
    stats = (
        sc
        .wholeTextFiles(args.output + '*.stats.json')
        .mapValues(json.loads)
        .map(lambda d: (d[0].replace('.stats.json', '').replace(args.output, ''), d[1]))
        .collectAsMap()
    )
    LOG.debug('all data: %s', stats)

    # pretty printing
    checkpoint_w = max(len(c) for c in stats.keys())
    table_divider = '-'
    checkpoint_title = 'Checkpoint'
    print(f'| {checkpoint_title: <{checkpoint_w}}   |'
          ' AP       | APM      | APL      |'
          ' t_{total} [ms]  | t_{dec} [ms] |'
          '     size |')
    print(f'|-{table_divider:{table_divider}<{checkpoint_w}}--:|'
          ':--------:|:--------:|:--------:|'
          ':---------------:|:------------:|'
          '---------:|')
    for checkpoint, data in sorted(stats.items(), key=lambda b_d: b_d[1]['stats'][0]):
        AP = 100.0 * data['stats'][0]
        APM = 100.0 * data['stats'][3]
        APL = 100.0 * data['stats'][4]
        t = 1000.0 * data['total_time'] / data['n_images']
        tdec = 1000.0 * data['decoder_time'] / data['n_images']
        file_size = data['file_size'] / 1024 / 1024
        print(
            f'| [{checkpoint: <{checkpoint_w}}] '
            f'| __{AP: <2.1f}__ '
            f'| {APM: <8.1f} '
            f'| {APL: <8.1f} '
            f'| {t: <15.0f} '
            f'| {tdec: <12.0f} '
            f'| {file_size: >6.1f}MB |'
        )


if __name__ == '__main__':
    main()
