"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import torch

from . import decoder, logger, network, show, visualizer, __version__
from .predictor import Predictor

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True,
                        help='Whether to output an image, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    return args


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg


def main():
    args = cli()
    annotation_painter = show.AnnotationPainter()

    predictor = Predictor(
        visualize_image=(args.show or args.image_output is not None),
        visualize_processed_image=args.debug,
    )
    for pred, _, meta in predictor.images(args.images):
        # json output
        if args.json_output is not None:
            json_out_name = out_name(
                args.json_output, meta['file_name'], '.predictions.json')
            LOG.debug('json output = %s', json_out_name)
            with open(json_out_name, 'w') as f:
                json.dump([ann.json_data() for ann in pred], f)

        # image output
        if args.show or args.image_output is not None:
            ext = show.Canvas.out_file_extension
            image_out_name = out_name(
                args.image_output, meta['file_name'], '.predictions.' + ext)
            LOG.debug('image output = %s', image_out_name)
            image = visualizer.Base.image()
            with show.image_canvas(image, image_out_name) as ax:
                annotation_painter.annotations(ax, pred)


if __name__ == '__main__':
    main()
