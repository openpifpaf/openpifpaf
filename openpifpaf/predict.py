"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import PIL
import torch

from . import datasets, decoder, logger, network, plugin, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)


# pylint: disable=too-many-statements
def cli():
    plugin.register()

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
    network.cli(parser)
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
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='rescale the long side of the image (aspect ratio maintained)')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    if args.debug_images:
        args.debug = True
    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    decoder.configure(args)
    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    return args


def processor_factory(args):
    # load model
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(head_metas)
    return processor, model


def preprocess_factory(args):
    rescale_t = None
    if args.long_edge:
        rescale_t = transforms.RescaleAbsolute(args.long_edge)

    pad_t = None
    if args.batch_size > 1:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        pad_t = transforms.CenterPad(args.long_edge)
    else:
        pad_t = transforms.CenterPadTight(16)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        pad_t,
        transforms.EVAL_TRANSFORM,
    ])


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

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = datasets.ImageList(args.images, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    annotation_painter = show.AnnotationPainter()

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            pred = preprocess.annotations_inverse(pred, meta)

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
            visualizer.Base.image(cpu_image)

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
                with show.image_canvas(cpu_image, image_out_name) as ax:
                    annotation_painter.annotations(ax, pred)


if __name__ == '__main__':
    main()
