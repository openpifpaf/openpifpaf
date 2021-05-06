"""CLI to convert a single-image checkpoint to a tracking checkpoint."""

import argparse
import logging
import os

import torch

import openpifpaf

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)

    openpifpaf.plugin.register()
    openpifpaf.logger.cli(parser)
    openpifpaf.network.Factory.cli(parser)

    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()

    openpifpaf.logger.configure(args, LOG)
    openpifpaf.network.Factory.configure(args)

    assert args.checkpoint, 'have to specify a checkpoint as input'
    if args.output is None:
        basename = 't' + os.path.basename(args.checkpoint)
        if not basename.endswith('.pkl'):
            basename += '-{}.pkl'.format(openpifpaf.__version__)
        args.output = os.path.join('outputs', basename)

    return args


def main():
    args = cli()
    model, _ = openpifpaf.network.Factory().factory()
    model.base_net = openpifpaf.network.TrackingBase(model.base_net)
    # model = openpifpaf.network.batchrenorm.BatchRenorm2d.convert_to(model)

    LOG.info('saving %s', args.output)
    torch.save({
        'model': model,
        'epoch': 0,
        'meta': {
            'image-source': args.checkpoint,
        },
    }, args.output)


if __name__ == '__main__':
    main()
