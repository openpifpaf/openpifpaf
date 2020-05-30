"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import PIL
import torch

from . import datasets, decoder, network, show, transforms, visualizer, __version__

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    visualizer.cli(parser)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--image-output', default=None, nargs='*',
                        help='image output file or directory')
    parser.add_argument('--json-output', default=None, nargs='*',
                        help='json output file or directory')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--line-width', default=6, type=int,
                        help='line width for skeleton')
    parser.add_argument('--monocolor-connections', default=False, action='store_true')
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig()
    logging.getLogger('openpifpaf').setLevel(log_level)
    LOG.setLevel(log_level)

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

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    return args


def processor_factory(args):
    # load model
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.head_names = model_cpu.head_names
        model.head_strides = model_cpu.head_strides
    processor = decoder.factory_from_args(args, model)
    return processor, model


def preprocess_factory(args):
    preprocess = [transforms.NormalizeAnnotations()]
    if args.long_edge:
        preprocess.append(transforms.RescaleAbsolute(args.long_edge))
    if args.batch_size > 1:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        preprocess.append(transforms.CenterPad(args.long_edge))
    else:
        preprocess.append(transforms.CenterPadTight(16))
    return transforms.Compose(preprocess + [transforms.EVAL_TRANSFORM])


def out_name(arg_list, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg_list can be:
    - none: return none (e.g. show image but don't store it)
    - empty: activate this output and determine a default name
    - one entry:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg_list is None:
        return None

    if len(arg_list) == 0:
        return in_name + default_extension

    if len(arg_list) == 1:
        oname = arg_list[0]
        if os.path.isdir(oname):
            oname = os.path.join(
                oname,
                os.path.basename(in_name)
            ) + default_extension
        return oname

    raise Exception('provide one or no value instead of {}'.format(arg_list))


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
    keypoint_painter = show.KeypointPainter(
        color_connections=not args.monocolor_connections,
        linewidth=args.line_width,
    )
    annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

            visualizer.BaseVisualizer.image(cpu_image)
            if preprocess is not None:
                pred = preprocess.annotations_inverse(pred, meta)

            if args.json_output is not None:
                json_out_name = out_name(
                    args.json_output, meta['file_name'], '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in pred], f)

            if args.show or args.image_output is not None:
                image_out_name = out_name(
                    args.image_output, meta['file_name'], '.predictions.png')
                LOG.debug('image output = %s', image_out_name)
                with show.image_canvas(cpu_image,
                                       image_out_name,
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    annotation_painter.annotations(ax, pred)


if __name__ == '__main__':
    main()
