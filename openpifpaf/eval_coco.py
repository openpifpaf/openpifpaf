"""Evaluation on COCO data."""

import argparse
import json
import logging
import os
import time
import zipfile

import numpy as np
import torch

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

from .data import COCO_PERSON_SKELETON
from .network import nets
from . import datasets, decoder, encoder, show, transforms

ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'
ANNOTATIONS_TESTDEV = 'data-mscoco/annotations/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data-mscoco/annotations/image_info_test2017.json'
IMAGE_DIR_TEST = 'data-mscoco/images/test2017/'

LOG = logging.getLogger(__name__)


class EvalCoco(object):
    def __init__(self, coco, processor, annotations_inverse, *,
                 max_per_image=20, small_threshold=0.0):
        self.coco = coco
        self.processor = processor
        self.annotations_inverse = annotations_inverse
        self.max_per_image = max_per_image
        self.small_threshold = small_threshold

        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.decoder_time = 0.0

    def stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        cat_ids = self.coco.getCatIds(catNms=['person'])
        print('cat_ids', cat_ids)

        coco_eval = self.coco.loadRes(predictions)

        self.eval = COCOeval(self.coco, coco_eval, iouType='keypoints')
        self.eval.params.catIds = cat_ids

        if image_ids is not None:
            print('image ids', image_ids)
            self.eval.params.imgIds = image_ids
        self.eval.evaluate()
        self.eval.accumulate()
        self.eval.summarize()
        return self.eval.stats

    @staticmethod
    def view_keypoints(image_cpu, annotations, gt):
        highlight = [5, 7, 9, 11, 13, 15]
        keypoint_painter = show.KeypointPainter(highlight=highlight)
        skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                                markersize=1, linewidth=6)

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            keypoint_painter.annotations(ax, [ann for ann in annotations if ann.score() > 0.01])

        with show.canvas() as ax:
            ax.set_axis_off()
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            skeleton_painter.annotations(ax, [ann for ann in annotations if ann.score() > 0.01])

        instances_gt = None
        if gt:
            instances_gt = np.stack([a['keypoints'] for a in gt])

            # for test: overwrite prediction with true values
            # instances = instances_gt.copy()[:1]

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            keypoint_painter.keypoints(ax, instances_gt, skeleton=COCO_PERSON_SKELETON)

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            show.white_screen(ax)
            keypoint_painter.keypoints(ax, instances_gt, color='lightgrey',
                                       skeleton=COCO_PERSON_SKELETON)
            keypoint_painter.annotations(ax, [ann for ann in annotations if ann.score() > 0.01])

    def from_predictions(self, predictions, meta,
                         debug=False, gt=None, image_cpu=None, verbose=False,
                         category_id=1):
        image_id = int(meta['image_id'])
        self.image_ids.append(image_id)

        if debug:
            self.view_keypoints(image_cpu, predictions, gt)

        predictions = self.annotations_inverse(predictions, meta)
        if self.small_threshold:
            predictions = [pred for pred in predictions
                           if pred.scale(v_th=0.01) >= self.small_threshold]
        if len(predictions) > self.max_per_image:
            predictions = predictions[:self.max_per_image]
        image_annotations = []
        for pred in predictions:
            # avoid visible keypoints becoming invisible due to rounding
            v_mask = pred.data[:, 2] > 0.0
            pred.data[v_mask, 2] = np.maximum(0.01, pred.data[v_mask, 2])

            keypoints = np.around(pred.data, 2)
            keypoints[:, 2] = 2.0
            image_annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': keypoints.reshape(-1).tolist(),
                'score': max(0.01, pred.score()),
            })

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            image_annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': np.zeros((17*3,)).tolist(),
                'score': 0.01,
            })

        if debug:
            self.stats(image_annotations, [image_id])
            if verbose:
                print('detected', image_annotations, len(image_annotations))
                oks = self.eval.computeOks(image_id, category_id)
                oks[oks < 0.5] = 0.0
                print('oks', oks)
                print('evaluate', self.eval.evaluateImg(image_id, category_id, (0, 1e5 ** 2), 20))
            print(meta)

        self.predictions += image_annotations

    def write_predictions(self, filename):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        print('wrote {}'.format(filename + '.pred.json'))
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        print('wrote {}'.format(filename + '.zip'))


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
    if args.two_scale:
        output += '-twoscale'
    if args.multi_scale:
        output += '-multiscale'
        if args.multi_scale_hflip:
            output += 'whflip'

    return output


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=True)
    encoder.cli(parser)
    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
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
                        help='long edge of input images')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
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
    args = parser.parse_args()

    logging.basicConfig()
    log_level = logging.INFO if not args.debug else logging.DEBUG
    logging.getLogger('openpifpaf').setLevel(log_level)
    LOG.setLevel(log_level)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    if args.dataset == 'val':
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = ANNOTATIONS_VAL
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

    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    return args


def write_evaluations(eval_coco, filename, args, total_time):
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
                'total_time': total_time,
                'checkpoint': args.checkpoint,
            }, f)
    else:
        print('given dataset does not have ground truth, so no stats summary')

    print('n images = {}'.format(n_images))
    print('decoder time = {:.1f}s ({:.0f}ms / image)'
          ''.format(eval_coco.decoder_time, 1000 * eval_coco.decoder_time / n_images))
    print('total time = {:.1f}s ({:.0f}ms / image)'
          ''.format(total_time, 1000 * total_time / n_images))


def preprocess_factory_from_args(args):
    collate_fn = datasets.collate_images_anns_meta
    if args.batch_size == 1 and not args.multi_scale:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])
    else:
        preprocess = transforms.Compose([
            transforms.NormalizeAnnotations(),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
            transforms.EVAL_TRANSFORM,
        ])

    return preprocess, collate_fn


def main():
    args = cli()

    # skip existing?
    if args.skip_existing:
        if os.path.exists(args.output + '.stats.json'):
            print('Output file {} exists already. Exiting.'
                  ''.format(args.output + '.stats.json'))
            return
        print('Processing: {}'.format(args.checkpoint))

    preprocess, collate_fn = preprocess_factory_from_args(args)
    data = datasets.CocoKeypoints(
        root=args.image_dir,
        annFile=args.annotation_file,
        preprocess=preprocess,
        all_persons=True,
        all_images=args.all_images,
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, pin_memory=args.pin_memory,
        num_workers=args.loader_workers, collate_fn=collate_fn)

    model_cpu, _ = nets.factory_from_args(args)
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.head_names = model_cpu.head_names
        model.head_strides = model_cpu.head_strides

    processor = decoder.factory_from_args(args, model, args.device)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    coco = pycocotools.coco.COCO(args.annotation_file)
    eval_coco = EvalCoco(coco, processor, preprocess.annotations_inverse)
    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors_cpu, anns_batch, meta_batch) in enumerate(data_loader):
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

        fields_batch = processor.fields(image_tensors_cpu)

        decoder_start = time.perf_counter()
        pred_batch = processor.annotations_batch(
            fields_batch, meta_batch=meta_batch, debug_images=image_tensors_cpu)
        eval_coco.decoder_time += time.perf_counter() - decoder_start

        # loop over batch
        assert len(image_tensors_cpu) == len(fields_batch)
        assert len(image_tensors_cpu) == len(anns_batch)
        assert len(image_tensors_cpu) == len(meta_batch)
        for image_tensor_cpu, pred, anns, meta in zip(
                image_tensors_cpu, pred_batch, anns_batch, meta_batch):
            eval_coco.from_predictions(pred, meta,
                                       debug=args.debug, gt=anns,
                                       image_cpu=image_tensor_cpu)
    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')
    write_evaluations(eval_coco, args.output, args, total_time)


if __name__ == '__main__':
    main()
