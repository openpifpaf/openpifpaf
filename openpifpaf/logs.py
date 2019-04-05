"""Configuring and visualizing log files."""

import argparse
import datetime
import json
import logging
from pprint import pprint

import numpy as np
import pysparkling

from . import show


def cli(parser):
    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')


def configure(args):
    from pythonjsonlogger import jsonlogger
    import socket
    import sys
    from . import __version__ as VERSION

    file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(
        jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG,
                        handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })


class Plots(object):
    def __init__(self, log_files, labels=None, output_prefix=None):
        self.log_files = log_files
        self.datas = [self.read_log(f) for f in log_files]
        self.labels = labels or log_files
        self.output_prefix = output_prefix or log_files[-1] + '.'

    @staticmethod
    def read_log(path):
        sc = pysparkling.Context()
        return (sc
                .textFile(path)
                .filter(lambda line: line.startswith(('{', 'json:')) and line.endswith('}'))
                .map(lambda line: json.loads(line.strip('json:')))
                .groupBy(lambda data: data.get('type'))
                .collectAsMap())

    def process(self):
        return {label: data['process']
                for data, label in zip(self.datas, self.labels)}

    def process_arguments(self):
        return {label: data['process'][0]['argv'][1:]
                for data, label in zip(self.datas, self.labels)}

    def time(self, ax):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = [datetime.datetime.strptime(row.get('asctime')[:-4], '%Y-%m-%d %H:%M:%S')
                     for row in data['train']]
                y = [(yi - y[0]).total_seconds() / 3600.0 for yi in y]
                ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('time [h]')
        ax.legend()

    def epoch_time(self, ax):
        for data, label in zip(self.datas, self.labels):
            y0 = None
            if 'train' in data:
                row = data['train'][0]
                y0 = datetime.datetime.strptime(row.get('asctime')[:-4], '%Y-%m-%d %H:%M:%S')
            if 'train-epoch' in data:
                x = [row.get('epoch') for row in data['train-epoch']]
                y = [datetime.datetime.strptime(row.get('asctime')[:-4], '%Y-%m-%d %H:%M:%S')
                     for row in data['train-epoch']]
                if y0 is not None:
                    x = [0] + x
                    y = [y0] + y
                y = [(yi - prev_yi).total_seconds() / 60.0
                     for prev_yi, yi in zip(y[:-1], y[1:])]
                ax.plot(x[1:], y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('epoch-time [min]')
        ax.legend()

    def lr(self, ax):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                x = [row.get('epoch') for row in data['train']]
                y = [row.get('lr') for row in data['train']]
                ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_yscale('log', nonposy='clip')
        ax.legend()

    def epoch_loss(self, ax):
        for data, label in zip(self.datas, self.labels):
            val_color = None
            if 'val-epoch' in data:
                x = np.array([row.get('epoch') for row in data['val-epoch']])
                y = np.array([row.get('loss') for row in data['val-epoch']], dtype=np.float)
                val_line, = ax.plot(x, y, 'o-', markersize=2, label=label)
                val_color = val_line.get_color()
            if 'train-epoch' in data:
                x = np.array([row.get('epoch') for row in data['train-epoch']])
                y = np.array([row.get('loss') for row in data['train-epoch']], dtype=np.float)
                m = x > 0
                ax.plot(x[m], y[m], 'x-', color=val_color, linestyle='dotted', markersize=2)

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        # ax.set_ylim(0.0, 4.0)
        # if min(y) > -0.1:
        #     ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        ax.legend()

    def epoch_head(self, ax, head_i):
        for data, label in zip(self.datas, self.labels):
            val_color = None
            if 'val-epoch' in data:
                x = np.array([row.get('epoch') for row in data['val-epoch']])
                y = np.array([row.get('head_losses')[head_i]
                              for row in data['val-epoch']], dtype=np.float)
                val_line, = ax.plot(x, y, 'o-', markersize=2, label=label)
                val_color = val_line.get_color()
            if 'train-epoch' in data:
                x = np.array([row.get('epoch') for row in data['train-epoch']])
                y = np.array([row.get('head_losses')[head_i]
                              for row in data['train-epoch']], dtype=np.float)
                m = x > 0
                ax.plot(x[m], y[m], 'x-', color=val_color, linestyle='dotted', markersize=2)

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss, head {}'.format(head_i))
        # ax.set_ylim(0.0, 1.0)
        # if min(y) > -0.1:
        #     ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        # ax.legend()

    def preprocess_time(self, ax):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = np.array([row.get('data_time') / row.get('time') * 100.0
                              for row in data['train']], dtype=np.float)
                stride = int(len(x) / (x[-1] - x[0]) / 30.0)  # 30 per epoch
                if stride > 5 and len(x) / stride > 2:
                    x_binned = np.array([x[i] for i in range(0, len(x), stride)][:-1])
                    y_binned = np.stack([y[i:i + stride] for i in range(0, len(x), stride)][:-1])
                    y_mean = np.mean(y_binned, axis=1)
                    y_min = np.min(y_binned, axis=1)
                    y_max = np.max(y_binned, axis=1)
                    line, = ax.plot(x_binned, y_mean, label=label)
                    ax.fill_between(x_binned, y_min, y_max, alpha=0.2,
                                    facecolor=line.get_color())
                else:
                    ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('data preprocessing time [%]')
        ax.set_ylim(0, 100)
        ax.legend()

    def train(self, ax):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = np.array([row.get('loss')
                              for row in data['train']], dtype=np.float)
                stride = int(len(x) / (x[-1] - x[0]) / 30.0)  # 30 per epoch
                if stride > 5 and len(x) / stride > 2:
                    x_binned = np.array([x[i] for i in range(0, len(x), stride)][:-1])
                    y_binned = np.stack([y[i:i + stride] for i in range(0, len(x), stride)][:-1])
                    y_mean = np.mean(y_binned, axis=1)
                    y_min = np.min(y_binned, axis=1)
                    y_max = np.max(y_binned, axis=1)
                    line, = ax.plot(x_binned, y_mean, label=label)
                    ax.fill_between(x_binned, y_min, y_max, alpha=0.2,
                                    facecolor=line.get_color())
                else:
                    ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('training loss')
        # ax.set_ylim(0, 8)
        if min(y) > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        ax.legend()

    def train_head(self, ax, head_i):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                x = np.array([row.get('epoch') + row.get('batch') / row.get('n_batches')
                              for row in data['train']])
                y = np.array([row.get('head_losses')[head_i]
                              for row in data['train']], dtype=np.float)
                m = np.logical_not(np.isnan(y))
                x, y = x[m], y[m]
                stride = int(len(x) / (x[-1] - x[0]) / 30.0)  # 30 per epoch
                if stride > 5 and len(x) / stride > 2:
                    x_binned = np.array([x[i] for i in range(0, len(x), stride)][:-1])
                    y_binned = np.stack([y[i:i + stride] for i in range(0, len(x), stride)][:-1])
                    y_mean = np.mean(y_binned, axis=1)
                    y_min = np.min(y_binned, axis=1)
                    y_max = np.max(y_binned, axis=1)
                    line, = ax.plot(x_binned, y_mean, label=label)
                    ax.fill_between(x_binned, y_min, y_max, alpha=0.2,
                                    facecolor=line.get_color())
                else:
                    ax.plot(x, y, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('training loss, head {}'.format(head_i))
        ax.set_ylim(3e-3, 3.0)
        if min(y) > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        # ax.legend()

    def print_last_line(self):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                print('{}: {}'.format(label, data['train'][-1]))

    def show_all(self, n_heads=5, *, share_y=True):
        pprint(self.process_arguments())

        with show.canvas() as ax:
            self.time(ax)

        with show.canvas() as ax:
            self.epoch_time(ax)

        with show.canvas() as ax:
            self.lr(ax)

        with show.canvas(ncols=n_heads, figsize=(20, 5), sharey=share_y) as axs:
            for i, ax in enumerate(axs):
                self.epoch_head(ax, -n_heads + i)

        with show.canvas() as ax:
            self.epoch_loss(ax)

        # with show.canvas() as ax:
        #     self.preprocess_time(ax)

        with show.canvas(ncols=n_heads, figsize=(20, 5), sharey=share_y) as axs:
            for i, ax in enumerate(axs):
                self.train_head(ax, -n_heads + i)

        with show.canvas() as ax:
            self.train(ax)

        self.print_last_line()


class EvalPlots(object):
    def __init__(self, log_files, labels=None, output_prefix=None,
                 edge=321, samples=200, decoder=0):
        self.edge = edge
        self.samples = samples
        self.decoder = decoder

        self.datas = [self.read_log(f) for f in log_files]
        self.labels = labels or log_files
        self.output_prefix = output_prefix or log_files[-1] + '.'

    def read_log(self, path):
        sc = pysparkling.Context()

        # modify individual file names and comma-seperated filenames
        files = path.split(',')
        files = ','.join(
            '{}.epoch???.evalcoco-edge{}-samples{}-decoder{}.txt'
            ''.format(f[:-4], self.edge, self.samples, self.decoder)
            for f in files
        )

        def epoch_from_filename(filename):
            i = filename.find('epoch')
            return int(filename[i+5:i+8])

        return (sc
                .wholeTextFiles(files)
                .map(lambda k_c: (
                    epoch_from_filename(k_c[0]),
                    [float(l) for l in k_c[1].splitlines()],
                ))
                .filter(lambda k_c: len(k_c[1]) == 10)
                .sortByKey()
                .collect())

    def frame(self, ax, entry):
        for data, label in zip(self.datas, self.labels):
            if not data:
                continue
            x = np.array([e for e, _ in data])
            y = np.array([d[entry] for _, d in data])
            ax.plot(x, y, 'o-', label=label, markersize=2)

        ax.set_xlabel('epoch')
        ax.grid(linestyle='dotted')
        # ax.legend()

    def ap(self, ax):
        self.frame(ax, entry=0)
        ax.set_ylabel('AP')

    def ap050(self, ax):
        self.frame(ax, entry=1)
        ax.set_ylabel('AP$^{0.50}$')

    def ap075(self, ax):
        self.frame(ax, entry=2)
        ax.set_ylabel('AP$^{0.75}$')

    def apm(self, ax):
        self.frame(ax, entry=3)
        ax.set_ylabel('AP$^{M}$')

    def apl(self, ax):
        self.frame(ax, entry=4)
        ax.set_ylabel('AP$^{L}$')

    def ar(self, ax):
        self.frame(ax, entry=5)
        ax.set_ylabel('AR')

    def ar050(self, ax):
        self.frame(ax, entry=6)
        ax.set_ylabel('AR$^{0.50}$')

    def ar075(self, ax):
        self.frame(ax, entry=7)
        ax.set_ylabel('AR$^{0.75}$')

    def arm(self, ax):
        self.frame(ax, entry=8)
        ax.set_ylabel('AR$^{M}$')

    def arl(self, ax):
        self.frame(ax, entry=9)
        ax.set_ylabel('AR$^{L}$')

    def fill_all(self, axs):
        for f, ax in zip((self.ap, self.ap050, self.ap075, self.apm, self.apl), axs[0]):
            f(ax)

        for f, ax in zip((self.ar, self.ar050, self.ar075, self.arm, self.arl), axs[1]):
            f(ax)

        return self

    def show_all(self, *, share_y=True):
        with show.canvas(nrows=2, ncols=5, figsize=(20, 10),
                         sharex=True, sharey=share_y) as axs:
            self.fill_all(axs)
            axs[0, 4].legend(fontsize=5)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('log_file', nargs='+',
                        help='path to log file')
    parser.add_argument('--label', nargs='+',
                        help='labels in the same order as files')
    parser.add_argument('--n-heads', default=6, type=int,
                        help='number of head losses')
    parser.add_argument('--eval-edge', default=593, type=int,
                        help='side length during eval')
    parser.add_argument('--eval-samples', default=200, type=int,
                        help='number of samples during eval')
    parser.add_argument('--no-share-y', dest='share_y',
                        default=True, action='store_false',
                        help='dont share y access')
    parser.add_argument('-o', '--output', default=None,
                        help='output prefix (default is log_file + .)')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.log_file[-1] + '.'

    EvalPlots(args.log_file, args.label, args.output,
              edge=args.eval_edge,
              samples=args.eval_samples).show_all(share_y=args.share_y)
    Plots(args.log_file, args.label, args.output).show_all(
        args.n_heads, share_y=args.share_y)


if __name__ == '__main__':
    main()
