"""Configuring and visualizing log files."""

import argparse
from collections import defaultdict
import datetime
import json
import logging
from pprint import pprint

import numpy as np
import pysparkling

from . import metric, show, __version__

try:
    import matplotlib
except ImportError:
    matplotlib = None

LOG = logging.getLogger(__name__)


def optionally_shaded(ax, x, y, *, color, label, **kwargs):
    stride = int(len(x) / (x[-1] - x[0]) / 30.0) if len(x) > 30 else 1  # 30 per epoch
    if stride > 1:
        x_binned = np.array([x[i] for i in range(0, len(x), stride)][:-1])
        y_binned = np.stack([y[i:i + stride] for i in range(0, len(x), stride)][:-1])
        y_mean = np.mean(y_binned, axis=1)
        y_min = np.min(y_binned, axis=1)
        y_max = np.max(y_binned, axis=1)
        ax.plot(x_binned, y_mean, color=color, label=label, **kwargs)
        ax.fill_between(x_binned, y_min, y_max, alpha=0.2, facecolor=color)
    else:
        LOG.debug('not shading: entries = %d, epochs = %f', len(x), x[-1] - x[0])
        ax.plot(x, y, color=color, label=label, **kwargs)


def fractional_epoch(row, *, default=None):
    """Given a data row, compute the fractional epoch taking batch into account.

    Example:
        Epoch 1 at batch 30 out of 100 batches per epoch would return
        epoch 1.3.
    """

    if 'epoch' not in row:
        return default
    if 'batch' not in row:
        return row.get('epoch')
    return row.get('epoch') + row.get('batch') / row.get('n_batches')


class Plots():
    def __init__(self, log_files, labels=None, *,
                 output_prefix=None, first_epoch=0.0, share_y=True):
        self.log_files = log_files
        self.labels = labels or [lf.replace('outputs/', '') for lf in log_files]
        self.output_prefix = output_prefix or log_files[-1] + '.'
        self.first_epoch = first_epoch
        self.share_y = share_y

        self.datas = [self.read_log(f) for f in log_files]

    def read_log(self, path):
        sc = pysparkling.Context()
        return (sc
                .textFile(path)
                .filter(lambda line: line.startswith(('{', 'json:')) and line.endswith('}'))
                .map(lambda line: json.loads(line.strip('json:')))
                .filter(lambda data: fractional_epoch(data, default=np.inf) >= self.first_epoch)
                .groupBy(lambda data: data.get('type'))
                .collectAsMap())

    def process(self):
        return {label: data['process']
                for data, label in zip(self.datas, self.labels)}

    def field_names(self):
        def migrate(field_name):
            """to process older (pre v0.12) log files"""
            if field_name.startswith(('cif.', 'caf.', 'caf25.')):
                return 'cocokp.{}'.format(field_name)
            if field_name.startswith('cifdet.'):
                return 'cocodet.{}'.format(field_name)
            return field_name
        return {label: [migrate(f) for f in data['config'][0]['field_names']]
                for data, label in zip(self.datas, self.labels)}

    def process_arguments(self):
        if not self.datas[0]:
            raise Exception('no data')
        return {label: data['process'][0]['argv'][1:]
                for data, label in zip(self.datas, self.labels)}

    def time(self, ax):
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

            if 'train' in data:
                x = np.array([fractional_epoch(row) for row in data['train']])
                y = [datetime.datetime.strptime(row.get('asctime')[:-4], '%Y-%m-%d %H:%M:%S')
                     for row in data['train']]
                y = [(yi - y[0]).total_seconds() / 3600.0 for yi in y]
                ax.plot(x, y, color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('time [h]')
        ax.legend(loc='upper left')

    def epoch_time(self, ax):
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

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
                ax.plot(x[1:], y, color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('epoch-time [min]')
        ax.legend(loc='lower right')

    def lr(self, ax):
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

            if 'train' in data:
                x = [fractional_epoch(row) for row in data['train']]
                y = [row.get('lr') for row in data['train']]
                ax.plot(x, y, color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('learning rate')
        ax.set_yscale('log', nonposy='clip')
        ax.legend(loc='upper left')

    def epoch_loss(self, ax):
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

            if 'val-epoch' in data:
                x = np.array([row.get('epoch') for row in data['val-epoch']])
                y = np.array([row.get('loss') for row in data['val-epoch']], dtype=np.float)
                ax.plot(x, y, 'o-', color=color, markersize=2, label=label)

            if 'train-epoch' in data:
                x = np.array([row.get('epoch') for row in data['train-epoch']])
                y = np.array([row.get('loss') for row in data['train-epoch']], dtype=np.float)
                m = x > 0
                ax.plot(x[m], y[m], 'x-', color=color, linestyle='dotted', markersize=2)

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        # ax.set_ylim(0.0, 4.0)
        # if min(y) > -0.1:
        #     ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        ax.legend(loc='upper right')
        ax.text(0.01, 1.01, 'train (cross-dotted), val (dot-solid)',
                transform=ax.transAxes, size='x-small')

    def epoch_head(self, ax, field_name):
        field_names = self.field_names()
        last_five_y = []
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)
            if field_name not in field_names[label]:
                continue
            field_i = field_names[label].index(field_name)
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

            if 'val-epoch' in data:
                x = np.array([row.get('epoch') for row in data['val-epoch']])
                y = np.array([row.get('head_losses')[field_i]
                              for row in data['val-epoch']], dtype=np.float)
                m = np.logical_not(np.isnan(y))
                ax.plot(x[m], y[m], 'o-', color=color, markersize=2, label=label)
                last_five_y.append(y[m][-5:])

            if 'train-epoch' in data:
                x = np.array([row.get('epoch') for row in data['train-epoch']])
                y = np.array([row.get('head_losses')[field_i]
                              for row in data['train-epoch']], dtype=np.float)
                m = np.logical_not(np.isnan(y))
                ax.plot(x[m], y[m], 'x-', color=color, linestyle='dotted', markersize=2)
                last_five_y.append(y[m][-5:])

        if not last_five_y:
            return
        ax.set_xlabel('epoch')
        ax.set_ylabel(field_name)
        last_five_y = np.concatenate(last_five_y)
        if not self.share_y and last_five_y.shape[0]:
            ax.set_ylim(np.min(last_five_y), np.max(last_five_y))
        # ax.set_ylim(0.0, 1.0)
        # if min(y) > -0.1:
        #     ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        # ax.legend(loc='upper right')
        ax.text(0.01, 1.01, 'train (cross-dotted), val (dot-solid)',
                transform=ax.transAxes, size='x-small')

    def preprocess_time(self, ax):
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)

            if 'train' in data:
                # skip batch 0 as it has corrupted data_time
                x = np.array([fractional_epoch(row)
                              for row in data['train']
                              if row.get('batch', 1) > 0])
                y = np.array([row.get('data_time') / row.get('time') * 100.0
                              for row in data['train']
                              if row.get('batch', 1) > 0], dtype=np.float)
                optionally_shaded(ax, x, y, color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel('data preprocessing time [%]')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')

    def train(self, ax):
        miny = 0.0
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)
            if 'train' in data:
                xy_all = defaultdict(list)
                for row in data['train']:
                    xy_all[row.get('loss_index', 0)].append(
                        (fractional_epoch(row), row.get('loss'))
                    )
                for loss_index, xy in xy_all.items():
                    x = np.array([x for x, _ in xy])
                    y = np.array([y for _, y in xy], dtype=np.float)
                    miny = min(miny, np.min(y))

                    kwargs = {}
                    this_label = label
                    if loss_index != 0:
                        kwargs['alpha'] = 0.5
                        this_label = '{} ({})'.format(label, loss_index)
                    optionally_shaded(ax, x, y, color=color, label=this_label, **kwargs)

        ax.set_xlabel('epoch')
        ax.set_ylabel('training loss')
        # ax.set_ylim(0, 8)
        if miny > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        ax.legend(loc='upper right')

    def train_head(self, ax, field_name):
        field_names = self.field_names()
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)
            if field_name not in field_names[label]:
                continue
            field_i = field_names[label].index(field_name)

            if 'train' in data:
                x = np.array([fractional_epoch(row) for row in data['train']])
                y = np.array([row.get('head_losses')[field_i]
                              for row in data['train']], dtype=np.float)
                m = np.logical_not(np.isnan(y))
                optionally_shaded(ax, x[m], y[m], color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel(format(field_name))
        # ax.set_ylim(3e-3, 3.0)
        if not self.share_y and min(y) > -0.1:
            ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        # ax.legend(loc='upper right')

    def mtl_sigma(self, ax, field_name):
        field_names = self.field_names()
        for color_i, (data, label) in enumerate(zip(self.datas, self.labels)):
            color = matplotlib.cm.get_cmap('tab10')((color_i % 10 + 0.05) / 10)
            if field_name not in field_names[label]:
                continue
            field_i = field_names[label].index(field_name)

            if 'train' in data:
                x = np.array([fractional_epoch(row) for row in data['train']])
                y = np.array([row['mtl_sigmas'][field_i] if 'mtl_sigmas' in row else np.nan
                              for row in data['train']], dtype=np.float)
                m = np.logical_not(np.isnan(y))
                optionally_shaded(ax, x[m], y[m], color=color, label=label)

        ax.set_xlabel('epoch')
        ax.set_ylabel(field_name)
        ax.set_ylim(-0.1, 1.1)
        if min(y) > -0.1:
            ax.set_ylim(3e-3, 3.0)
            ax.set_yscale('log', nonposy='clip')
        ax.grid(linestyle='dotted')
        # ax.legend(loc='upper right')

    def print_last_line(self):
        for data, label in zip(self.datas, self.labels):
            if 'train' in data:
                print('{}: {}'.format(label, data['train'][-1]))

    def show_all(self, show_mtl_sigmas=False):
        pprint(self.process_arguments())

        all_field_names = [f for fs in self.field_names().values() for f in fs]
        rows = defaultdict(list)
        for f in all_field_names:
            dataset_name, head_name = f.split('.')[:2]
            row_name = dataset_name + '.' + head_name
            if f not in rows[row_name]:
                rows[row_name].append(f)
        n_rows = len(rows)
        n_cols = max(len(r) for r in rows.values())
        multi_figsize = (5 * n_cols, 2.5 * n_rows)

        with show.canvas() as ax:
            self.time(ax)

        with show.canvas() as ax:
            self.epoch_time(ax)

        with show.canvas() as ax:
            self.lr(ax)

        with show.canvas(nrows=n_rows, ncols=n_cols, squeeze=False,
                         figsize=multi_figsize,
                         sharey=self.share_y, sharex=True) as axs:
            for row_i, row in enumerate(rows.values()):
                for col_i, field_name in enumerate(row):
                    self.epoch_head(axs[row_i, col_i], field_name)

        with show.canvas() as ax:
            self.epoch_loss(ax)

        with show.canvas() as ax:
            self.preprocess_time(ax)

        with show.canvas(nrows=n_rows, ncols=n_cols, squeeze=False,
                         figsize=multi_figsize,
                         sharey=self.share_y, sharex=True) as axs:
            for row_i, row in enumerate(rows.values()):
                for col_i, field_name in enumerate(row):
                    self.train_head(axs[row_i, col_i], field_name)

        if show_mtl_sigmas:
            with show.canvas(nrows=n_rows, ncols=n_cols, squeeze=False,
                             figsize=multi_figsize,
                             sharey=self.share_y, sharex=True) as axs:
                for row_i, row in enumerate(rows.values()):
                    for col_i, field_name in enumerate(row):
                        self.mtl_sigma(axs[row_i, col_i], field_name)

        with show.canvas() as ax:
            self.train(ax)

        self.print_last_line()


class EvalPlots():
    #: Eval files come with text labels. This is a translation to prettier labels
    #: for matplotlib axes.
    text_to_latex_labels = {
        'AP0.5': 'AP$^{0.50}$',
        'AP0.75': 'AP$^{0.75}$',
        'APS': 'AP$^{S}$',
        'APM': 'AP$^{M}$',
        'APL': 'AP$^{L}$',
        'ART1': 'AR@1',
        'ART10': 'AR@10',
        'AR0.5': 'AR$^{0.50}$',
        'AR0.75': 'AR$^{0.75}$',
        'ARS': 'AR$^{S}$',
        'ARM': 'AR$^{M}$',
        'ARL': 'AR$^{L}$',
    }

    def __init__(self, log_files, file_suffix, *,
                 labels=None, output_prefix=None,
                 decoder=0, legend_last_ap=True,
                 first_epoch=0.0, share_y=True):
        self.file_suffix = file_suffix
        self.decoder = decoder
        self.legend_last_ap = legend_last_ap
        self.first_epoch = first_epoch
        self.share_y = share_y

        self.datas = [self.read_log(f) for f in log_files]
        self.labels = labels or [lf.replace('outputs/', '') for lf in log_files]
        self.output_prefix = output_prefix or log_files[-1] + '.'

    def read_log(self, path):
        sc = pysparkling.Context()

        # modify individual file names and comma-seperated filenames
        files = path.split(',')
        files = ','.join([
            '{}.epoch???{}'.format(f[:-4], self.file_suffix)
            for f in files
        ])

        def epoch_from_filename(filename):
            i = filename.find('epoch')
            return int(filename[i + 5:i + 8])

        def migrate(data):
            # earlier versions did not contain 'dataset'
            if 'dataset' not in data and len(data['stats']) == 10:
                data['dataset'] = 'cocokp'
            if 'dataset' not in data and len(data['stats']) == 12:
                data['dataset'] = 'cocodet'

            # earlier versions did not contain 'text_labels'
            if 'text_labels' not in data and len(data['stats']) == 10:
                data['text_labels'] = metric.Coco.text_labels_keypoints
            if 'text_labels' not in data and len(data['stats']) == 12:
                data['text_labels'] = metric.Coco.text_labels_bbox

            return data

        return (sc
                .wholeTextFiles(files)
                .map(lambda k_c: (
                    epoch_from_filename(k_c[0]),
                    json.loads(k_c[1]),
                ))
                .filter(lambda k_c: k_c[0] >= self.first_epoch and k_c[1]['stats'])
                .mapValues(migrate)
                .sortByKey()
                .collect())

    def metrics(self):
        all_metrics_by_datasets = defaultdict(list)
        for data in self.datas:
            if not data:
                continue
            dataset = data[0][1]['dataset']
            for m in data[0][1]['text_labels']:
                if m in all_metrics_by_datasets[dataset]:
                    continue
                all_metrics_by_datasets[dataset].append(m)
        return all_metrics_by_datasets

    def fill_metric(self, ax, dataset, metric_name):
        for data, label in zip(self.datas, self.labels):
            if not data:
                continue
            if data[0][1]['dataset'] != dataset:
                continue
            if metric_name not in data[0][1]['text_labels']:
                continue

            entry = data[0][1]['text_labels'].index(metric_name)
            if self.legend_last_ap:
                last_main_value = data[-1][1]['stats'][0]
                main_name = data[0][1]['text_labels'][0]
                main_label = self.text_to_latex_labels.get(main_name, main_name)
                label = '{} ({}={:.1%})'.format(label, main_label, last_main_value)
            x = np.array([e for e, _ in data])
            y = np.array([d['stats'][entry] for _, d in data])
            ax.plot(x, y, 'o-', label=label, markersize=2)

        ax.set_xlabel('epoch')
        ax.set_ylabel('{} {}'.format(
            dataset,
            self.text_to_latex_labels.get(metric_name, metric_name),
        ))
        ax.grid(linestyle='dotted')
        # ax.legend(loc='upper right')

    def frame_ops(self, ax, entry):
        assert entry in (0, 1)

        s = 1e9 if entry == 0 else 1e6
        for data, label in zip(self.datas, self.labels):
            if not data:
                continue
            x = np.array([d.get('count_ops', [0, 0])[entry] / s for _, d in data])[-1]
            if x == 0.0:
                continue
            y = np.array([d['stats'][0] for _, d in data])[-1]
            ax.plot([x], [y], 'o', label=label, markersize=10)
            ax.annotate(
                label if len(label) < 20 else label.split('-')[0],
                (x, y),
                xytext=(0.0, -5.0),
                textcoords='offset points',
                rotation=90,
                horizontalalignment='center', verticalalignment='top',
            )

        # ax.set_ylim(bottom=0.56)
        ax.set_xlabel('GMACs' if entry == 0 else 'million parameters')
        ax.set_ylabel('AP')
        ax.grid(linestyle='dotted')
        # ax.legend(loc='lower right')

    def show_all(self):
        # layouting: a dataset can span one or two rows
        all_metrics = self.metrics()
        all_rows_nested = {
            dataset: (
                [metrics]
                if len(metrics) <= 6
                else [metrics[:-len(metrics) // 2],
                      metrics[-len(metrics) // 2:]]
            )
            for dataset, metrics in all_metrics.items()
        }
        all_rows = [
            [(dataset, metric) for metric in row]
            for dataset, rows in all_rows_nested.items()
            for row in rows
        ]
        if not all_rows:
            return
        nrows = len(all_rows)
        ncols = max(len(row) for row in all_rows)

        # plot
        with show.canvas(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows),
                         sharex=True, sharey=self.share_y) as axs:
            for ax_row, metric_row in zip(axs, all_rows):
                for ax, (dataset, metric_name) in zip(ax_row, metric_row):
                    self.fill_metric(ax, dataset, metric_name)
                ax_row[len(metric_row) - 1].legend(fontsize=5, loc='lower right')

        with show.canvas(nrows=1, ncols=2, figsize=(10, 5),
                         sharey=self.share_y) as axs:
            self.frame_ops(axs[0], 0)
            self.frame_ops(axs[1], 1)


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.logs',
        usage='%(prog)s [options] log_files',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    show.cli(parser)

    parser.add_argument('log_file', nargs='+',
                        help='path to log file(s)')
    parser.add_argument('--label', nargs='+',
                        help='label(s) in the same order as files')
    parser.add_argument('--eval-suffix', default='.eval-cocokp.stats.json',
                        help='suffix of evaluation files to look for')
    parser.add_argument('--first-epoch', default=1e-6, type=float,
                        help='epoch (can be float) of first data point to plot')
    parser.add_argument('--no-share-y', dest='share_y',
                        default=True, action='store_false',
                        help='dont share y access')
    parser.add_argument('-o', '--output', default=None,
                        help='output prefix (default is log_file + .)')
    parser.add_argument('--show-mtl-sigmas', default=False, action='store_true')
    args = parser.parse_args()

    show.configure(args)

    if args.output is None:
        args.output = args.log_file[-1] + '.'

    EvalPlots(args.log_file, args.eval_suffix,
              labels=args.label,
              output_prefix=args.output,
              first_epoch=args.first_epoch,
              share_y=args.share_y,
              ).show_all()
    Plots(args.log_file, args.label,
          output_prefix=args.output,
          first_epoch=args.first_epoch,
          share_y=args.share_y,
          ).show_all(show_mtl_sigmas=args.show_mtl_sigmas)


if __name__ == '__main__':
    main()
