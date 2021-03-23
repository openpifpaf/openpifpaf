"""
python -m openpifpaf.predict_pan --checkpoint chlpt.epoch\
 test_cvsports_dataset/images_trainvaltest/test/* --output output

"""

import argparse
import glob
import json
import logging
import os
import numpy as np
import imageio

import logging
from collections import OrderedDict
import os
import json

import numpy as np
from tabulate import tabulate

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
    parser.add_argument('--image-output', default=None, nargs='?', const=True,
                        help='image output file or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
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
    group.add_argument('--output', required=True)
    group.add_argument('--skip-pred', default=False, action='store_true')
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
    # print('Head nets')
    # print(model.head_nets)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        # print('in if processor')
        # print(model_cpu)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets
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
    # data_loader = torch.utils.data.DataLoader(
    #     data, batch_size=args.batch_size, shuffle=False,
    #     pin_memory=args.pin_memory, num_workers=args.loader_workers,
    #     collate_fn=datasets.collate_images_anns_meta)

    ### AMA
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_targets_inst_meta)

    # visualizers
    keypoint_painter = show.KeypointPainter(
        color_connections=not args.monocolor_connections,
        linewidth=args.line_width,
    )
    # print('HERERERER',processor)
    if isinstance(processor, decoder.CifCent):  # Check if cifcent decoder
        # print('yyyyyyyyyyyyyyyyyyyyyyy')
        # show.A
        keypoint_painter = show.KeypointCentPainter(
            color_connections=not args.monocolor_connections,
            linewidth=args.line_width,
        )

    annotations = []

    if not args.skip_pred:
        annotation_painter = show.AnnotationPainter(keypoint_painter=keypoint_painter)
        for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
            pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

            # unbatch
            for image, pred, meta in zip(image_tensors_batch, pred_batch, meta_batch):
                #print('meta')
                #print(meta)
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
                        # print('PRED',[ann.json_data() for ann in pred])

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

                # eval_coco.from_predictions(pred, meta, debug=args.debug, gt=anns)
                segments = []
                panoptic = np.zeros((image.shape[1:3]), np.uint16)

                n_humans = 0
                for ann in pred:
                    if ann.category_id == 1 and ann.mask.any():
                        n_humans += 1
                        instance_id = 1000+n_humans
                        panoptic[ann.mask] = instance_id

                        segments.append({
                            'id': instance_id,
                            'category_id': 1
                        })
                image_id = meta['file_name'].split('/')[-1].replace('.png', '')
                panoptic_name = 'output/%s.png'%image_id

                background = panoptic == 0
                if background.any():
                    panoptic[background] = 2000
                    segments.append({
                        'id': 2000,
                        'category_id': 2
                    })

                annotations.append({
                    'image_id': image_id,
                    'file_name': panoptic_name,
                    'segments_info': segments
                })


                panoptic = np.stack([
                    panoptic.astype(np.uint8),
                    (panoptic//256).astype(np.uint8),
                    (panoptic//256//256).astype(np.uint8)
                ], axis=-1)
                imageio.imwrite(panoptic_name, panoptic)

    gt_json_file = 'test_cvsports_dataset/keemotion_panoptic_test.json'
    gt_folder = 'test_cvsports_dataset/annotations_trainvaltest/test/'
    pred_json_file = '%s.json'%args.output
    pred_folder = './'#'%s'%args.output

    if not args.skip_pred:
        struct = json.load(open(gt_json_file, 'r'))
        struct['annotations'] = annotations
        json.dump(struct, open(pred_json_file, 'w'))

    evaluate(gt_json_file, gt_folder, pred_json_file, pred_folder)



def evaluate(gt_json_file, gt_folder, pred_json_file, pred_folder):
    from panopticapi.evaluation import pq_compute

    pq_res = pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder)

    res = {}
    res["PQ"] = 100 * pq_res["All"]["pq"]
    res["SQ"] = 100 * pq_res["All"]["sq"]
    res["RQ"] = 100 * pq_res["All"]["rq"]
    res["PQ_th"] = 100 * pq_res["Things"]["pq"]
    res["SQ_th"] = 100 * pq_res["Things"]["sq"]
    res["RQ_th"] = 100 * pq_res["Things"]["rq"]
    res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
    res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
    res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

    results = OrderedDict({"panoptic_seg": res})
    LOG.info(results)
    _print_panoptic_results(pq_res)

    return results

def _print_panoptic_results(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    LOG.info("Panoptic Evaluation Results:\n" + table)

if __name__ == '__main__':
    main()
