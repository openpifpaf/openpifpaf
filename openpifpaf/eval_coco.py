"""Evaluation on COCO data."""

import argparse
import json
import logging
import os
import time
import zipfile

import numpy as np
import torch

import pycocotools.coco
from pycocotools.cocoeval import COCOeval

from .data import COCO_PERSON_SKELETON
from .network import nets
from . import datasets, decoder, encoder, show, transforms

ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'
ANNOTATIONS_TESTDEV = 'data-mscoco/annotations/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data-mscoco/annotations/image_info_test2017.json'
IMAGE_DIR_TEST = 'data-mscoco/images/test2017/'

# monkey patch for Python 3 compat
pycocotools.coco.unicode = str


class EvalCoco(object):
    def __init__(self, coco, processor, keypoint_sets_inverse, skeleton=None):
        self.coco = coco
        self.processor = processor
        self.keypoint_sets_inverse = keypoint_sets_inverse
        self.skeleton = skeleton or COCO_PERSON_SKELETON

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

    def view_keypoints(self, image_cpu, annotations, gt):
        highlight = [5, 7, 9, 11, 13, 15]
        keypoint_painter = show.KeypointPainter(skeleton=self.skeleton, highlight=highlight)
        skeleton_painter = show.KeypointPainter(skeleton=self.skeleton,
                                                show_box=False, color_connections=True,
                                                markersize=1, linewidth=6)

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            keypoint_painter.annotations(ax, annotations)

        with show.canvas() as ax:
            ax.set_axis_off()
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            skeleton_painter.annotations(ax, [ann for ann in annotations if ann.score() > 0.1])

        instances_gt = None
        if gt:
            instances_gt = np.stack([a['keypoints'] for a in gt])

            # for test: overwrite prediction with true values
            # instances = instances_gt.copy()[:1]

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            keypoint_painter.keypoints(ax, instances_gt)

        with show.canvas() as ax:
            ax.imshow((np.moveaxis(image_cpu.numpy(), 0, -1) + 2.0) / 4.0)
            show.white_screen(ax)
            keypoint_painter.keypoints(ax, instances_gt, color='lightgrey')
            keypoint_painter.annotations(ax, [ann for ann in annotations if ann.score() > 0.01])

    def from_fields(self, fields, meta,
                    debug=False, gt=None, image_cpu=None, verbose=False,
                    category_id=1):
        if image_cpu is not None:
            self.processor.set_cpu_image(None, image_cpu)

        start = time.time()
        annotations = self.processor.annotations(fields, meta)[:20]
        self.decoder_time += time.time() - start

        if isinstance(meta, (list, tuple)):
            meta = meta[0]

        image_id = int(meta['image_id'])
        self.image_ids.append(image_id)

        if debug:
            self.view_keypoints(image_cpu, annotations, gt)

        instances, scores = self.processor.keypoint_sets_from_annotations(annotations)
        instances = self.keypoint_sets_inverse(instances, meta)
        image_annotations = []
        for instance, score in zip(instances, scores):
            keypoints = np.around(instance, 2)
            keypoints[:, 2] = 2.0
            image_annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': keypoints.reshape(-1).tolist(),
                'score': score,
            })

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            image_annotations.append({
                'image_id': image_id,
                'category_id': category_id,
                'keypoints': np.zeros((17*3,)).tolist(),
                'score': 0.0,
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
        with open(filename + '.json', 'w') as f:
            json.dump(predictions, f)
        print('wrote {}'.format(filename + '.json'))
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.json', arcname='predictions.json')
        print('wrote {}'.format(filename + '.zip'))


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    decoder.cli(parser, force_complete_pose=True)
    encoder.cli(parser)
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
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--two-scale', default=False, action='store_true',
                        help='two scale')
    parser.add_argument('--three-scale', default=False, action='store_true',
                        help='three scale')
    parser.add_argument('--multi-scale', default=False, action='store_true',
                        help='multi scale')
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

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    if args.dataset == 'val':
        image_dir = IMAGE_DIR_VAL
        annotation_file = ANNOTATIONS_VAL
    elif args.dataset == 'test':
        image_dir = IMAGE_DIR_TEST
        annotation_file = ANNOTATIONS_TEST
    elif args.dataset == 'test-dev':
        image_dir = IMAGE_DIR_TEST
        annotation_file = ANNOTATIONS_TESTDEV
    else:
        raise Exception

    if args.dataset in ('test', 'test-dev') and not args.write_predictions:
        raise Exception('have to use --write-predictions for this dataset')
    if args.dataset in ('test', 'test-dev') and not args.all_images:
        raise Exception('have to use --all-images for this dataset')

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args, image_dir, annotation_file


def write_evaluations(eval_cocos, args):
    for i, eval_coco in enumerate(eval_cocos):
        filename = '{}.evalcoco-{}edge{}-samples{}-{}decoder{}'.format(
            args.checkpoint,
            '{}-'.format(args.dataset) if args.dataset != 'val' else '',
            args.long_edge, args.n,
            'noforcecompletepose-' if not args.force_complete_pose else '', i)

        if args.write_predictions:
            eval_coco.write_predictions(filename)

        if args.dataset not in ('test', 'test-dev'):
            stats = eval_coco.stats()
            np.savetxt(filename + '.txt', stats)
        else:
            print('given dataset does not have ground truth, so no stats summary')

        print('Decoder {}: decoder time = {}s'.format(i, eval_coco.decoder_time))


def preprocess_factory_from_args(args):
    collate_fn = datasets.collate_images_anns_meta
    if args.two_scale:
        preprocess = transforms.MultiScale([
            transforms.Normalize(),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute(args.long_edge),
            ]),
        ])
        collate_fn = datasets.collate_multiscale_images_anns_meta
    elif args.three_scale:
        preprocess = transforms.MultiScale([
            transforms.Normalize(),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleRelative(2.0),
            ]),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute(args.long_edge),
            ]),
        ])
        collate_fn = datasets.collate_multiscale_images_anns_meta
    elif args.multi_scale:
        preprocess = transforms.MultiScale([
            transforms.RescaleAbsolute((args.long_edge - 1) * 4 + 1),
            transforms.RescaleAbsolute((args.long_edge - 1) * 3 + 1),
            transforms.RescaleAbsolute((args.long_edge - 1) * 2 + 1),
            transforms.RescaleAbsolute(args.long_edge),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute(args.long_edge),
            ]),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute((args.long_edge - 1) * 2 + 1),
            ]),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute((args.long_edge - 1) * 3 + 1),
            ]),
            transforms.Compose([
                transforms.HFlip(),
                transforms.RescaleAbsolute((args.long_edge - 1) * 4 + 1),
            ]),
        ])
        collate_fn = datasets.collate_multiscale_images_anns_meta
    elif args.batch_size == 1:
        preprocess = transforms.RescaleAbsolute(args.long_edge)
    else:
        preprocess = transforms.Compose([
            transforms.RescaleAbsolute(args.long_edge),
            transforms.CenterPad(args.long_edge),
        ])

    return preprocess, collate_fn


def main():
    args, image_dir, annotation_file = cli()

    # skip existing?
    eval_output_filename = '{}.evalcoco-edge{}-samples{}-decoder{}.txt'.format(
        args.checkpoint, args.long_edge, args.n, 0)
    if args.skip_existing:
        if os.path.exists(eval_output_filename):
            print('Output file {} exists already. Exiting.'.format(eval_output_filename))
            return
        print('Processing: {}'.format(args.checkpoint))

    preprocess, collate_fn = preprocess_factory_from_args(args)
    data = datasets.CocoKeypoints(
        root=image_dir,
        annFile=annotation_file,
        preprocess=preprocess,
        all_persons=True,
        all_images=args.all_images,
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, pin_memory=args.pin_memory,
        num_workers=args.loader_workers, collate_fn=collate_fn)

    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    processor = decoder.factory_from_args(args, model, args.device)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    coco = pycocotools.coco.COCO(annotation_file)
    eval_coco = EvalCoco(coco, processor, preprocess.keypoint_sets_inverse)
    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors_cpu, anns_batch, meta_batch) in enumerate(data_loader):
        logging.info('batch %d, last loop: %.3fs, batches per second=%.1f',
                     batch_i, time.time() - loop_start,
                     batch_i / max(1, (time.time() - total_start)))
        if batch_i < args.skip_n:
            continue
        if args.n and batch_i >= args.n:
            break

        loop_start = time.time()

        # detect multiscale
        multiscale = isinstance(image_tensors_cpu, list)
        if multiscale:
            # only look at first scale
            anns_batch = anns_batch[0]

        if len([a
                for anns in anns_batch
                for a in anns
                if np.any(a['keypoints'][:, 2] > 0)]) < args.min_ann:
            continue

        fields_batch = processor.fields(image_tensors_cpu)

        if multiscale:
            # only look at first scale
            image_tensors_cpu = image_tensors_cpu[0]

        # loop over batch
        assert len(image_tensors_cpu) == len(fields_batch)
        assert len(image_tensors_cpu) == len(anns_batch)
        assert len(image_tensors_cpu) == len(meta_batch)
        for image_tensor_cpu, fields, anns, meta in zip(
                image_tensors_cpu, fields_batch, anns_batch, meta_batch):
            if args.debug and multiscale:
                for scale_i, (f, m) in enumerate(zip(fields, meta)):
                    print('scale', scale_i)
                    eval_coco.from_fields(f, m,
                                          debug=args.debug, gt=anns, image_cpu=image_tensor_cpu)

            eval_coco.from_fields(fields, meta,
                                  debug=args.debug, gt=anns, image_cpu=image_tensor_cpu)
    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')
    write_evaluations([eval_coco], args)
    print('total processing time = {}s'.format(total_time))


if __name__ == '__main__':
    main()
