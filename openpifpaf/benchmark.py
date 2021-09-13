"""Benchmark."""

import argparse
from collections import namedtuple
from dataclasses import dataclass
import datetime
import json
import logging
import os
import subprocess
from typing import List

import pysparkling
try:
    import tabulate
except ImportError:
    tabulate = None

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
    parser.add_argument('--v012-ablation-1', default=False, action='store_true')
    parser.add_argument('--v012-ablation-2', default=False, action='store_true')
    parser.add_argument('--v012-ablation-3', default=False, action='store_true')
    parser.add_argument('--v012-ablation-4', default=False, action='store_true')
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
        if not any(l.startswith('--decoder') for l in eval_args):
            LOG.info('adding "--decoder=cifcaf:0" to the argument list')
            eval_args.append('--decoder=cifcaf:0')

    # generate a default output filename
    if args.output is None:
        now = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        args.output = 'outputs/benchmark-{}/'.format(now)
        os.makedirs(args.output)

    return args, eval_args


@dataclass
class Config:
    checkpoint: str
    suffix: str
    args: List[str]

    @property
    def name(self):
        return (self.checkpoint + self.suffix).replace('/', '-')


class Benchmark:
    def __init__(self, configs, output_folder, *,
                 reference_config=None, stat_filter=None, stat_scale=1.0):
        assert tabulate is not None, 'need tabulate: pip install tabulate'
        self.configs = configs
        self.output_folder = output_folder
        self.reference_config = reference_config
        self.stat_filter = stat_filter
        self.stat_scale = stat_scale

    def run(self):
        for config in self.configs:
            self.run_config(config)
        return self

    def run_config(self, config: Config):
        out_file = os.path.join(self.output_folder, config.name)
        if os.path.exists(out_file + '.stats.json'):
            LOG.warning('Output file %s exists already. Skipping.',
                        out_file + '.stats.json')
            return

        LOG.debug('Launching eval for %s.', config.name)
        cmd = [
            'python', '-m', 'openpifpaf.eval',
            '--output', out_file,
            '--checkpoint', config.checkpoint,
        ] + config.args
        LOG.info('eval command: %s', ' '.join(cmd))
        subprocess.run(cmd, check=True)

        # print intermediate output
        self.print_md()

    def stats(self):
        sc = pysparkling.Context()
        stats = (
            sc
            .wholeTextFiles(self.output_folder + '*.stats.json')
            .mapValues(json.loads)
            .map(lambda d: (d[0].replace('.stats.json', '').replace(self.output_folder, ''), d[1]))
            .collectAsMap()
        )
        LOG.debug('all data: %s', stats)
        return stats

    def stat_values(self, stat):
        return [
            v * self.stat_scale
            for l, v in zip(stat['text_labels'], stat['stats'])
            if self.stat_filter is None or l in self.stat_filter
        ]

    def stat_text_labels(self, stat):
        return [
            l
            for l in stat['text_labels']
            if self.stat_filter is None or l in self.stat_filter
        ]

    def print_md(self):
        """Pretty printing markdown"""
        stats = self.stats()
        first_stats = list(stats.values())[0]

        headers = ['Name']
        headers += self.stat_text_labels(first_stats)
        headers += ['t_total', 't_NN', 't_dec', 'size']

        reference_values = None
        if self.reference_config is not None:
            reference = stats.get(self.reference_config.name)
            if reference is not None:
                reference_values = self.stat_values(reference)

        table = []
        for name, data in sorted(stats.items(), key=lambda b_d: self.stat_values(b_d[1])[0]):
            values = self.stat_values(data)
            t = 1000.0 * data['total_time'] / data['n_images']
            tnn = 1000.0 * data['nn_time'] / data['n_images']
            tdec = 1000.0 * data['decoder_time'] / data['n_images']
            file_size = data['file_size'] / 1024 / 1024

            row = [f'[{name}]']

            if reference_values is not None and self.reference_config.name != name:
                values = [v - r for v, r in zip(values, reference_values)]
                row += [f'__{values[0]:+.1f}__']
                if len(values) > 1:
                    row += [f'{v:+.1f}' for v in values[1:]]

                t -= 1000.0 * reference['total_time'] / reference['n_images']
                tdec -= 1000.0 * reference['decoder_time'] / reference['n_images']
                tnn -= 1000.0 * reference['nn_time'] / reference['n_images']
                file_size -= reference['file_size'] / 1024 / 1024
                row += [f'{t:+.0f}ms', f'{tnn:+.0f}ms', f'{tdec:+.0f}ms', f'{file_size:+.1f}MB']
            else:
                row += [f'__{values[0]:.1f}__']
                if len(values) > 1:
                    row += [f'{v:.1f}' for v in values[1:]]
                row += [f'{t:.0f}ms', f'{tnn:.0f}ms', f'{tdec:.0f}ms', f'{file_size:.1f}MB']

            table.append(row)

        colalign = ['left'] + ['right'] * (len(headers) - 1)
        print(tabulate.tabulate(table, headers=headers, tablefmt='pipe', colalign=colalign))
        return self


def main():
    args, eval_args = cli()
    Ablation = namedtuple('Ablation', ['suffix', 'args'])
    ablations = [Ablation('', eval_args)]

    if args.iccv2019_ablation:
        ablations += [
            Ablation('.singlescale-max', eval_args + ['--connection-method=max']),
            Ablation('.singlescale', eval_args + ['--connection-method=blend']),
            Ablation('.multiscale-nohflip', eval_args + ['--connection-method=blend',
                                                         '--long-edge=961',
                                                         '--multi-scale',
                                                         '--no-multi-scale-hflip']),
            Ablation('.multiscale', eval_args + ['--connection-method=blend',
                                                 '--long-edge=961',
                                                 '--multi-scale']),
        ]
    if args.v012_ablation_1:
        ablations += [
            Ablation('.greedy', eval_args + ['--greedy']),
            Ablation('.no-reverse', eval_args + ['--no-reverse-match']),
            Ablation('.greedy.no-reverse', eval_args + ['--greedy', '--no-reverse-match']),
            Ablation('.greedy.dense', eval_args + [
                '--greedy', '--cocokp-with-dense', '--dense-connections']),
            Ablation('.dense', eval_args + [
                '--cocokp-with-dense', '--dense-connections']),
            Ablation('.dense.hierarchy', eval_args + [
                '--cocokp-with-dense', '--dense-connections=0.1']),
        ]
    if args.v012_ablation_2:
        ablations += [
            Ablation('.cifnr', eval_args + ['--ablation-cifseeds-no-rescore']),
            Ablation('.cifnr.nms', eval_args + ['--ablation-cifseeds-no-rescore',
                                                '--ablation-cifseeds-nms']),
            Ablation('.cafnr', eval_args + ['--ablation-caf-no-rescore']),
            Ablation('.nr.nms', eval_args + ['--ablation-cifseeds-no-rescore',
                                             '--ablation-cifseeds-nms',
                                             '--ablation-caf-no-rescore']),
        ]
    if args.v012_ablation_3:
        eval_args_nofc = [a for a in eval_args if not a.startswith('--force-complete')]
        ablations += [
            Ablation('.nofc', eval_args_nofc),
            Ablation('.nr.nms.nofc', eval_args_nofc + ['--ablation-cifseeds-no-rescore',
                                                       '--ablation-cifseeds-nms',
                                                       '--ablation-caf-no-rescore']),
        ]
    if args.v012_ablation_4:
        ablations += [
            Ablation('.indkp', eval_args + ['--ablation-independent-kp',
                                            '--keypoint-threshold=0.2']),
        ]

    configs = [
        Config(checkpoint, ablation.suffix, ablation.args)
        for checkpoint in args.checkpoints
        for ablation in ablations
    ]
    Benchmark(
        configs, args.output,
        reference_config=configs[0] if len(args.checkpoints) == 1 else None,
        stat_filter=('AP', 'AP0.5', 'AP0.75', 'APM', 'APL'),
        stat_scale=100.0,
    ).run()


if __name__ == '__main__':
    main()
