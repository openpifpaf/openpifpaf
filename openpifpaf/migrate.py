"""Migrate a model."""

import argparse
import logging

import torch

from . import network, __version__


def main():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.migrate',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--output')
    network.cli(parser)
    args = parser.parse_args()

    network.configure(args)

    if args.checkpoint is None:
        raise Exception('checkpoint must be provided for old model to migrate from')

    if args.output is None:
        args.output = args.checkpoint + '.out.pkl'

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # load old model
    checkpoint = torch.load(args.checkpoint)

    # create a new model
    args.checkpoint = None
    new_model, _ = network.factory_from_args(args)

    # transfer state from old to new model
    new_model.load_state_dict(checkpoint['model'].state_dict())
    checkpoint['model'] = new_model
    torch.save(checkpoint, args.output)


if __name__ == '__main__':
    main()
