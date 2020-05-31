"""Evaluation on COCO data."""

import argparse
import json
import logging
import os
import sys
import time
import zipfile

import numpy as np
import PIL
import thop
import torch

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

from .annotation import Annotation, AnnotationDet
from .datasets.constants import COCO_KEYPOINTS, COCO_PERSON_SKELETON, COCO_CATEGORIES
from . import datasets, decoder, network, show, transforms, visualizer, __version__

ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
DET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'
ANNOTATIONS_TESTDEV = 'data-mscoco/annotations/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data-mscoco/annotations/image_info_test2017.json'
IMAGE_DIR_TEST = 'data-mscoco/images/test2017/'

LOG = logging.getLogger(__name__)


class EvalCoco(object):
    def __init__(self, coco, processor, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0):
        if category_ids is None:
            category_ids = [1]

        self.coco = coco
        self.processor = processor
        self.max_per_image = max_per_image
        self.category_ids = category_ids
        self.iou_type = iou_type
        self.small_threshold = small_threshold

        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.decoder_time = 0.0
        self.nn_time = 0.0

        LOG.debug('max = %d, category ids = %s, iou_type = %s',
                  self.max_per_image, self.category_ids, self.iou_type)

    def stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        coco_eval = self.coco.loadRes(predictions)

        self.eval = COCOeval(self.coco, coco_eval, iouType=self.iou_type)
        LOG.info('cat_ids: %s', self.category_ids)
        if self.category_ids:
            self.eval.params.catIds = self.category_ids

        if image_ids is not None:
            print('image ids', image_ids)
            self.eval.params.imgIds = image_ids
        self.eval.evaluate()
        self.eval.accumulate()
        self.eval.summarize()
        return self.eval.stats

    @staticmethod
    def count_ops(model, height=641, width=641):
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, height, width, device=device)
        gmacs, params = thop.profile(model, inputs=(dummy_input, ))
        LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
        return gmacs, params

    @staticmethod
    def view_annotations(meta, predictions, ground_truth):
        annotation_painter = show.AnnotationPainter()
        with open(os.path.join(IMAGE_DIR_VAL, meta['file_name']), 'rb') as f:
            cpu_image = PIL.Image.open(f).convert('RGB')

        with show.image_canvas(cpu_image) as ax:
            annotation_painter.annotations(ax, predictions)

        if ground_truth:
            with show.image_canvas(cpu_image) as ax:
                show.white_screen(ax)
                annotation_painter.annotations(ax, ground_truth, color='grey')
                annotation_painter.annotations(ax, predictions)

    def from_predictions(self, predictions, meta, debug=False, gt=None):
        image_id = int(meta['image_id'])
        self.image_ids.append(image_id)

        predictions = transforms.Preprocess.annotations_inverse(predictions, meta)
        if self.small_threshold:
            predictions = [pred for pred in predictions
                           if pred.scale(v_th=0.01) >= self.small_threshold]
        if len(predictions) > self.max_per_image:
            predictions = predictions[:self.max_per_image]

        if debug:
            gt_anns = []
            for g in gt:
                if 'bbox' in g:
                    gt_anns.append(
                        AnnotationDet(COCO_CATEGORIES).set(g['category_id'] - 1, None, g['bbox'])
                    )
                if 'keypoints' in g:
                    gt_anns.append(
                        Annotation(COCO_KEYPOINTS, COCO_PERSON_SKELETON)
                        .set(g['keypoints'], fixed_score=None)
                    )
            gt_anns = transforms.Preprocess.annotations_inverse(gt_anns, meta)
            self.view_annotations(meta, predictions, gt_anns)

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            pred_data['image_id'] = image_id
            pred_data = {
                k: v for k, v in pred_data.items()
                if k in ('category_id', 'score', 'keypoints', 'bbox', 'image_id')
            }
            image_annotations.append(pred_data)

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            image_annotations.append({
                'image_id': image_id,
                'category_id': 1,
                'keypoints': np.zeros((17*3,)).tolist(),
                'bbox': [0, 0, 1, 1],
                'score': 0.001,
            })

        if debug:
            self.stats(image_annotations, [image_id])
            LOG.debug(meta)

        self.predictions += image_annotations

    def write_predictions(self, filename):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)


def default_output_name(args):
    output = '{}.evalcoco-{}edge{}'.format(
        args.checkpoint,
        '{}-'.format(args.dataset) if args.dataset != 'val' else '',
        args.long_edge,
    )
    if args.n:
        output += '-samples{}'.format(args.n)
    if not args.force_complete_pose:
        output += '-noforcecompletepose'
    if args.orientation_invariant or args.extended_scale:
        output += '-'
        if args.orientation_invariant:
            output += 'o'
        if args.extended_scale:
            output += 's'
    if args.two_scale:
        output += '-twoscale'
    if args.multi_scale:
        output += '-multiscale'
        if args.multi_scale_hflip:
            output += 'whflip'

    return output


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval_coco',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=True)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--detection-annotations', default=False, action='store_true')
    parser.add_argument('-n', default=0, type=int,
                        help='number of batches')
    parser.add_argument('--skip-n', default=0, type=int,
                        help='skip n batches')
    parser.add_argument('--dataset', choices=('val', 'test', 'test-dev'), default='val',
                        help='dataset to evaluate')
    parser.add_argument('--min-ann', default=0, type=int,
                        help='minimum number of truth annotations')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--long-edge', default=641, type=int,
                        help='long edge of input images. Setting to zero deactivates scaling.')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--orientation-invariant', default=False, action='store_true')
    parser.add_argument('--extended-scale', default=False, action='store_true')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--all-images', default=False, action='store_true',
                        help='run over all images irrespective of catIds')

    group = parser.add_argument_group('logging')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    group.add_argument('--debug-images', default=False, action='store_true',
                       help='print debug messages and enable all debug images')
    group.add_argument('--log-stats', default=False, action='store_true',
                       help='enable stats logging')

    args = parser.parse_args()

    if args.debug_images:
        args.debug = True

    log_level = logging.INFO if not args.debug else logging.DEBUG
    if args.log_stats:
        # pylint: disable=import-outside-toplevel
        from pythonjsonlogger import jsonlogger
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(
            jsonlogger.JsonFormatter('(message) (levelname) (name)'))
        logging.basicConfig(handlers=[stdout_handler])
        logging.getLogger('openpifpaf').setLevel(log_level)
        logging.getLogger('openpifpaf.stats').setLevel(logging.DEBUG)
        LOG.setLevel(log_level)
    else:
        logging.basicConfig()
        logging.getLogger('openpifpaf').setLevel(log_level)
        LOG.setLevel(log_level)

    network.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = max(2, args.batch_size)

    if args.dataset == 'val' and not args.detection_annotations:
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = ANNOTATIONS_VAL
    elif args.dataset == 'val' and args.detection_annotations:
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = DET_ANNOTATIONS_VAL
    elif args.dataset == 'test':
        args.image_dir = IMAGE_DIR_TEST
        args.annotation_file = ANNOTATIONS_TEST
    elif args.dataset == 'test-dev':
        args.image_dir = IMAGE_DIR_TEST
        args.annotation_file = ANNOTATIONS_TESTDEV
    else:
        raise Exception

    if args.dataset in ('test', 'test-dev') and not args.write_predictions and not args.debug:
        raise Exception('have to use --write-predictions for this dataset')
    if args.dataset in ('test', 'test-dev') and not args.all_images and not args.debug:
        raise Exception('have to use --all-images for this dataset')

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    return args


def write_evaluations(eval_coco, filename, args, total_time, count_ops, file_size):
    if args.write_predictions:
        eval_coco.write_predictions(filename)

    n_images = len(eval_coco.image_ids)

    if args.dataset not in ('test', 'test-dev'):
        stats = eval_coco.stats()
        np.savetxt(filename + '.txt', stats)
        with open(filename + '.stats.json', 'w') as f:
            json.dump({
                'stats': stats.tolist(),
                'n_images': n_images,
                'decoder_time': eval_coco.decoder_time,
                'nn_time': eval_coco.nn_time,
                'total_time': total_time,
                'checkpoint': args.checkpoint,
                'count_ops': count_ops,
                'file_size': file_size,
            }, f)
    else:
        print('given dataset does not have ground truth, so no stats summary')

    print('n images = {}'.format(n_images))
    print('decoder time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_coco.decoder_time, 1000 * eval_coco.decoder_time / n_images))
    print('nn time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_coco.nn_time, 1000 * eval_coco.nn_time / n_images))
    print('total time = {:.1f}s ({:.0f}ms / image)'
          ''.format(total_time, 1000 * total_time / n_images))


def preprocess_factory(
        long_edge,
        *,
        tight_padding=False,
        extended_scale=False,
        orientation_invariant=False,
):
    preprocess = [transforms.NormalizeAnnotations()]

    if extended_scale:
        assert long_edge
        preprocess += [
            transforms.DeterministicEqualChoice([
                transforms.RescaleAbsolute(long_edge),
                transforms.RescaleAbsolute((long_edge - 1) // 2 + 1),
            ], salt=1)
        ]
    elif long_edge:
        preprocess += [transforms.RescaleAbsolute(long_edge)]

    if tight_padding:
        preprocess += [transforms.CenterPadTight(16)]
    else:
        assert long_edge
        preprocess += [transforms.CenterPad(long_edge)]

    if orientation_invariant:
        preprocess += [
            transforms.DeterministicEqualChoice([
                None,
                transforms.RotateBy90(fixed_angle=90),
                transforms.RotateBy90(fixed_angle=180),
                transforms.RotateBy90(fixed_angle=270),
            ], salt=3)
        ]

    preprocess += [transforms.EVAL_TRANSFORM]
    return transforms.Compose(preprocess)


def dataloader_from_args(args):
    preprocess = preprocess_factory(
        args.long_edge,
        tight_padding=args.batch_size == 1 and not args.multi_scale,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
    )
    data = datasets.Coco(
        image_dir=args.image_dir,
        ann_file=args.annotation_file,
        preprocess=preprocess,
        image_filter='all' if args.all_images else 'annotated',
        category_ids=[] if args.detection_annotations else [1],
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    return data_loader


def main():
    args = cli()

    # skip existing?
    if args.skip_existing:
        if os.path.exists(args.output + '.stats.json'):
            print('Output file {} exists already. Exiting.'
                  ''.format(args.output + '.stats.json'))
            return
        print('Processing: {}'.format(args.checkpoint))

    data_loader = dataloader_from_args(args)
    model_cpu, _ = network.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets

    processor = decoder.factory_from_args(args, model)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    coco = pycocotools.coco.COCO(args.annotation_file)
    eval_coco = EvalCoco(
        coco,
        processor,
        max_per_image=100 if args.detection_annotations else 20,
        category_ids=[] if args.detection_annotations else [1],
        iou_type='bbox' if args.detection_annotations else 'keypoints',
    )
    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors, anns_batch, meta_batch) in enumerate(data_loader):
        LOG.info('batch %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        if batch_i < args.skip_n:
            continue
        if args.n and batch_i >= args.n:
            break

        loop_start = time.time()

        if len([a
                for anns in anns_batch
                for a in anns
                if np.any(a['keypoints'][:, 2] > 0)]) < args.min_ann:
            continue

        pred_batch = processor.batch(model, image_tensors, device=args.device)
        eval_coco.decoder_time += processor.last_decoder_time
        eval_coco.nn_time += processor.last_nn_time

        # loop over batch
        assert len(image_tensors) == len(anns_batch)
        assert len(image_tensors) == len(meta_batch)
        for pred, anns, meta in zip(pred_batch, anns_batch, meta_batch):
            eval_coco.from_predictions(pred, meta, debug=args.debug, gt=anns)
    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')

    # model stats
    count_ops = list(eval_coco.count_ops(model_cpu))
    local_checkpoint = network.local_checkpoint_path(args.checkpoint)
    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write
    write_evaluations(eval_coco, args.output, args, total_time, count_ops, file_size)


if __name__ == '__main__':
    main()
