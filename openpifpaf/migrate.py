"""Migrate a model."""

import argparse
import logging

import torch

from .network import nets


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--output')
    nets.cli(parser)
    args = parser.parse_args()

    if args.checkpoint is None:
        raise Exception('checkpoint must be provided for old model to migrate from')

    if args.output is None:
        args.output = args.checkpoint + '.out.pkl'

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # load old model
    checkpoint = torch.load(args.checkpoint)

    # create a new model
    args.checkpoint = None
    new_model, _ = nets.factory_from_args(args)

    # transfer state from old to new model
    new_model.load_state_dict(checkpoint['model'].state_dict())
    checkpoint['model'] = new_model
    torch.save(checkpoint, args.output)


if __name__ == '__main__':
    main()
