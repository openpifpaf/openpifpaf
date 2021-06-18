"""Benchmark."""

import argparse
from collections import namedtuple
import datetime
import logging
import os

import openpifpaf.benchmark

LOG = logging.getLogger(__name__)


DEFAULT_CHECKPOINTS = [
    'tshufflenetv2k16',
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
                        version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))

    parser.add_argument('--output', default=None,
                        help='output file name')
    parser.add_argument('--checkpoints', default=DEFAULT_CHECKPOINTS, nargs='+',
                        help='checkpoints to evaluate')
    parser.add_argument('--crowdpose', default=False, action='store_true')
    parser.add_argument('--ablation-1', default=False, action='store_true')
    parser.add_argument('--ablation-2', default=False, action='store_true')
    parser.add_argument('--ablation-3', default=False, action='store_true')
    parser.add_argument('--ablation-4', default=False, action='store_true')
    parser.add_argument('--ablation-5', default=False, action='store_true')
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args, eval_args = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # default eval_args
    if not eval_args:
        eval_args = ['--loader-workers=2']

    # default loader workers
    if not any(l.startswith('--loader-workers') for l in eval_args):
        LOG.info('adding "--loader-workers=2" to the argument list')
        eval_args.append('--loader-workers=2')

    # default dataset
    if not any(l.startswith('--dataset') for l in eval_args):
        if args.crowdpose:
            LOG.info('adding "--dataset=crowdpose" to the argument list')
            eval_args.append('--dataset=crowdpose')
            if not any(l.startswith('--force-complete-pose') for l in eval_args):
                LOG.info('adding "--force-complete-pose" to the argument list')
                eval_args.append('--force-complete-pose')
            if not any(l.startswith('--seed-threshold') for l in eval_args):
                LOG.info('adding "--seed-threshold=0.2" to the argument list')
                eval_args.append('--seed-threshold=0.2')
            if not any(l.startswith('--crowdpose-eval-test') for l in eval_args):
                LOG.info('adding "--crowdpose-eval-test" to the argument list')
                eval_args.append('--crowdpose-eval-test')
            if not any(l.startswith('--decoder') for l in eval_args):
                LOG.info('adding "--decoder=cifcaf:0" to the argument list')
                eval_args.append('--decoder=cifcaf:0')
        else:
            LOG.info('adding "--dataset=posetrack2018" to the argument list')
            eval_args.append('--dataset=posetrack2018')
            if not any(l.startswith('--write-predictions') for l in eval_args):
                LOG.info('adding "--write-predictions" to the argument list')
                eval_args.append('--write-predictions')
            if not any(l.startswith('--decoder') for l in eval_args):
                LOG.info('adding "--decoder=trackingpose:0" to the argument list')
                eval_args.append('--decoder=trackingpose:0')

    # generate a default output filename
    if args.output is None:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = 'outputs/benchmark-{}/'.format(now)
        os.makedirs(args.output)

    return args, eval_args


def main():
    args, eval_args = cli()
    Ablation = namedtuple('Ablation', ['suffix', 'args'])
    ablations = [Ablation('', eval_args)]

    if args.crowdpose:
        assert all('crowdpose' in c for c in args.checkpoints)
        ablations += [
            Ablation('.easy', eval_args + ['--crowdpose-index=easy']),
            Ablation('.medium', eval_args + ['--crowdpose-index=medium']),
            Ablation('.hard', eval_args + ['--crowdpose-index=hard']),
        ]

    if args.ablation_1:
        ablations += [
            Ablation('.greedy', eval_args + ['--greedy']),
            Ablation('.no-reverse', eval_args + ['--no-reverse-match']),
            Ablation('.greedy.no-reverse', eval_args + ['--greedy', '--no-reverse-match']),
            # Ablation('.greedy.dense', eval_args + ['--greedy', '--dense-connections']),
            # Ablation('.dense', eval_args + ['--dense-connections']),
            # Ablation('.dense.hierarchy', eval_args + ['--dense-connections=0.1']),
        ]
    if args.ablation_2:
        ablations += [
            Ablation('.nr.nms', eval_args + ['--ablation-cifseeds-no-rescore',
                                             '--ablation-cifseeds-nms',
                                             '--ablation-caf-no-rescore']),
        ]
    if args.ablation_3:
        eval_args_decabl = [
            arg
            for arg in eval_args
            if not arg.startswith(('--instance-threshold=', '--decoder='))
        ]
        ablations += [
            Ablation('.euclidean', eval_args_decabl + ['--decoder=posesimilarity:0',
                                                       '--posesimilarity-distance=euclidean']),
            Ablation('.oks', eval_args_decabl + ['--decoder=posesimilarity:0',
                                                 '--posesimilarity-distance=oks']),
            Ablation('.oks-inflate2', eval_args_decabl + ['--decoder=posesimilarity:0',
                                                          '--posesimilarity-distance=oks',
                                                          '--posesimilarity-oks-inflate=2.0']),
            Ablation('.oks-inflate10', eval_args_decabl + ['--decoder=posesimilarity:0',
                                                           '--posesimilarity-distance=oks',
                                                           '--posesimilarity-oks-inflate=10.0']),
        ]
    if args.ablation_4:
        ablations += [
            Ablation('.w513', eval_args + ['--posetrack-eval-long-edge=513']),
            Ablation('.w641', eval_args + ['--posetrack-eval-long-edge=641']),
            Ablation('.w1201', eval_args + ['--posetrack-eval-long-edge=1201']),
        ]
    if args.ablation_5:
        ablations += [
            Ablation('.recovery', eval_args + ['--trackingpose-track-recovery']),
        ]

    configs = [
        openpifpaf.benchmark.Config(checkpoint, ablation.suffix, ablation.args)
        for checkpoint in args.checkpoints
        for ablation in ablations
    ]
    openpifpaf.benchmark.Benchmark(
        configs,
        args.output,
        reference_config=configs[0] if len(args.checkpoints) == 1 and not args.crowdpose else None,
        stat_scale=100.0 if args.crowdpose else 1.0,
    ).run()


if __name__ == '__main__':
    main()
