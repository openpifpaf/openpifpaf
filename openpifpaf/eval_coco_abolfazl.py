"""Evaluation on COCO data."""

import argparse
from ast import parse
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
from . import datasets, encoder, decoder, network, show, transforms, visualizer, __version__

ANNOTATIONS_VAL = 'data-mscoco/annotations/person_keypoints_val2017.json'
DET_ANNOTATIONS_VAL = 'data-mscoco/annotations/instances_val2017.json'
IMAGE_DIR_VAL = 'data-mscoco/images/val2017/'
ANNOTATIONS_TESTDEV = 'data-mscoco/annotations/image_info_test-dev2017.json'
ANNOTATIONS_TEST = 'data-mscoco/annotations/image_info_test2017.json'
IMAGE_DIR_TEST = 'data-mscoco/images/test2017/'

ANNOTATIONS_VAL = '/scratch/abolfazl/coco/annotations/person_keypoints_val2017.json'
DET_ANNOTATIONS_VAL = '/scratch/abolfazl/coco/annotations/instances_val2017.json'
IMAGE_DIR_VAL = '/scratch/abolfazl/coco/images/val2017/'
ANNOTATIONS_TESTDEV = '/scratch/abolfazl/coco/annotations/image_info_test-dev2017.json'
ANNOTATIONS_TEST = '/scratch/abolfazl/coco/annotations/image_info_test2017.json'
IMAGE_DIR_TEST = '/scratch/abolfazl/coco/images/test2017/'

# ANNOTATIONS_VAL = '/data/mistasse/coco/annotations/person_keypoints_val2017.json'
# DET_ANNOTATIONS_VAL = '/data/mistasse/coco/annotations/instances_val2017.json'
# IMAGE_DIR_VAL = '/data/mistasse/coco/images/val2017/'
# ANNOTATIONS_TESTDEV = '/data/mistasse/coco/annotations/image_info_test-dev2017.json'
# ANNOTATIONS_TEST = '/data/mistasse/coco/annotations/image_info_test2017.json'
# IMAGE_DIR_TEST = '/data/mistasse/coco/images/test2017/'

LOG = logging.getLogger(__name__)

import panopticapi.evaluation as PQ
import datetime

OFFSET = 256 * 256 * 256
VOID = 0
categories_list = [{"id": 2, "name": "background", "color": [50, 50, 70], "supercategory": "void", "isthing": 0}, 
            {"id": 1, "name": "player", "color": [220, 20, 60], "supercategory": "human", "isthing": 1}, 
            {"id": 3, "name": "ball", "color": [20, 220, 60], "supercategory": "object", "isthing": 1}]

categories = {el['id']: el for el in categories_list}

class EvalCoco(object):
    def __init__(self, coco, processor, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0,
                 considerUnvisible=True,
                 mask_unvisible_keypoints=False,
                 target_keypoints=None):
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
        self.considerUnvisible = considerUnvisible
        self.mask_unvisible_keypoints = mask_unvisible_keypoints
        self.target_keypoints = target_keypoints

        LOG.debug('max = %d, category ids = %s, iou_type = %s',
                  self.max_per_image, self.category_ids, self.iou_type)

    def stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        coco_eval = self.coco.loadRes(predictions)

        self.eval = COCOeval(self.coco, coco_eval, iouType=self.iou_type, considerUnvisible=self.considerUnvisible, mask_unvisible_keypoints=self.mask_unvisible_keypoints, target_keypoints=self.target_keypoints)
        LOG.info('cat_ids: %s', self.category_ids)
        if self.category_ids:
            self.eval.params.catIds = self.category_ids

        if image_ids is not None:
            print('image ids', image_ids)
            self.eval.params.imgIds = image_ids

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~ run evaluate')
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
        prog='python3 -m openpifpaf.eval_coco_abolfazl',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=True)
    encoder.cli(parser)
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


    group.add_argument('--annotations-val', default=ANNOTATIONS_VAL)
    group.add_argument('--det-annotations-val', default=DET_ANNOTATIONS_VAL)
    group.add_argument('--image-dir-val', default=IMAGE_DIR_VAL)
    group.add_argument('--annotations-testdev', default=ANNOTATIONS_TESTDEV)
    group.add_argument('--annotations-test', default=ANNOTATIONS_TEST)
    group.add_argument('--image-dir-test', default=IMAGE_DIR_TEST)


    parser.add_argument('--oracle-data', default=None, nargs='+',
                        help='pass centroid, keypoints, semantic, and offset to use their oracle')

    parser.add_argument('--disable-pan-quality', default=False, action='store_true')
    parser.add_argument('--disable-json-results', default=False, action='store_true')

    parser.add_argument('--discard-smaller', default=0, type=int,
                        help='discard smaller than')

    parser.add_argument('--discard-lesskp', default=0, type=int,
                        help='discard with number of keypoints less than')

    parser.add_argument('--n-images', default=None, type=int)

    parser.add_argument('--only-visible-keypoints', default=False, action='store_true')
    parser.add_argument('--mask-unvisible-keypoints', default=False, action='store_true')

    group.add_argument('--list-mask-unvisible-keypoints', default=None, nargs='+',
                       help='list of keypoints you wish to remove unvisible ones of')
    
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

    # if args.dataset == 'val' and not args.detection_annotations:
    #     args.image_dir = IMAGE_DIR_VAL
    #     args.annotation_file = ANNOTATIONS_VAL
    # elif args.dataset == 'val' and args.detection_annotations:
    #     args.image_dir = IMAGE_DIR_VAL
    #     args.annotation_file = DET_ANNOTATIONS_VAL
    # elif args.dataset == 'test':
    #     args.image_dir = IMAGE_DIR_TEST
    #     args.annotation_file = ANNOTATIONS_TEST
    # elif args.dataset == 'test-dev':
    #     args.image_dir = IMAGE_DIR_TEST
    #     args.annotation_file = ANNOTATIONS_TESTDEV
    # else:
    #     raise Exception

    if args.dataset == 'val' and not args.detection_annotations:
        args.image_dir = IMAGE_DIR_VAL
        args.annotation_file = args.annotations_val # ANNOTATIONS_VAL
    elif args.dataset == 'val' and args.detection_annotations:
        args.image_dir = args.image_dir_val # IMAGE_DIR_VAL
        args.annotation_file = args.det_annotations_val #DET_ANNOTATIONS_VAL
    elif args.dataset == 'test':
        args.image_dir = args.image_dir_test #IMAGE_DIR_TEST
        args.annotation_file = args.annotations_test #ANNOTATIONS_TEST
    elif args.dataset == 'test-dev':
        args.image_dir = args.image_dir_test #IMAGE_DIR_TEST
        args.annotation_file = args.annotations_testdev #ANNOTATIONS_TESTDEV
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


def dataloader_from_args(args, target_transforms=None, heads=None):
    preprocess = preprocess_factory(
        args.long_edge,
        tight_padding=args.batch_size == 1 and not args.multi_scale,
        extended_scale=args.extended_scale,
        orientation_invariant=args.orientation_invariant,
    )
    heads_names = []
    for h in heads:
        heads_names.append(h.meta.name)
    config = 'cif'
    if 'cifball' in heads_names:
        config = 'cifball'
    elif 'cifcentball' in heads_names:
        config = 'cifcentball'
    elif 'cifcent' in heads_names:
        config = 'cifcent'
    if 'cif' in heads_names and 'cent' in heads_names:
        config = 'cif cent'
    print('CONFIG', config)
    # print('heads', heads[0].meta.name)
    data = datasets.Coco(
        image_dir=args.image_dir,
        ann_file=args.annotation_file,
        ann_inst_file=args.annotation_file,
        preprocess=preprocess,
        image_filter='all' if args.all_images else 'annotated',
        # image_filter='kp_inst',
        n_images=args.n_images,
        target_transforms=target_transforms,
        eval_coco=True,
        category_ids=[] if args.detection_annotations else [1],
        config=config,
        # config='pan' if args.checkpoint=='resnet50' else 'cif cent'
    )
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, pin_memory=args.pin_memory,
        num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_targets_inst_meta_eval,
        )

    return data_loader


def get_output_filename(args):
    import os
    if not os.path.exists('coco_logs'):
        os.makedirs('coco_logs')
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S.%f')
    out = 'coco_logs/{}-{}'.format(now, args.checkpoint.split('/')[-1])

    return out + '.log'

def get_json_filename(args, log_file_name):
    import os
    if not os.path.exists('coco_results'):
        os.makedirs('coco_results')
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S.%f')
    out = 'coco_results/{}-{}-LOG_{}'.format(now, args.checkpoint.split('/')[-1], log_file_name.split('-')[1])

    return out + '.json'


def pq_compute_single_core(panoptic_pred, pred_segments_info, panoptic_gt, gt_segments_info):
    pq_stat = PQ.PQStat()
    

    gt_segms = {el['id']: el for el in gt_segments_info}
    pred_segms = {el['id']: el for el in pred_segments_info}


    # predicted segments area calculation + prediction sanity checks
    pred_labels_set = set(el['id'] for el in pred_segments_info)
    labels, labels_cnt = np.unique(panoptic_pred, return_counts=True)
    for label, label_cnt in zip(labels, labels_cnt):
        if label not in pred_segms:
            if label == VOID:
                continue
            raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
        pred_segms[label]['area'] = label_cnt
        pred_labels_set.remove(label)
        if pred_segms[label]['category_id'] not in categories:
            raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
    if len(pred_labels_set) != 0:
        raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

    # confusion matrix calculation
    pan_gt_pred = panoptic_gt.astype(np.uint64) * OFFSET + panoptic_pred.astype(np.uint64)
    gt_pred_map = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    for label, intersection in zip(labels, labels_cnt):
        gt_id = label // OFFSET
        pred_id = label % OFFSET
        gt_pred_map[(gt_id, pred_id)] = intersection

    # count all matched pairs
    gt_matched = set()
    pred_matched = set()
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]['iscrowd'] == 1:
            continue
        if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
            continue

        union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
        iou = intersection / union
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]['category_id']].tp += 1
            pq_stat[gt_segms[gt_label]['category_id']].iou += iou
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)

    # count false positives
    crowd_labels_dict = {}
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        # crowd segments are ignored
        if gt_info['iscrowd'] == 1:
            crowd_labels_dict[gt_info['category_id']] = gt_label
            continue
        pq_stat[gt_info['category_id']].fn += 1

    # count false positives
    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        # intersection of the segment with VOID
        intersection = gt_pred_map.get((VOID, pred_label), 0)
        # plus intersection with corresponding CROWD region if it exists
        if pred_info['category_id'] in crowd_labels_dict:
            intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
        # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
        if intersection / pred_info['area'] > 0.5:
            continue
        pq_stat[pred_info['category_id']].fp += 1

    return pq_stat


def show_pq_results(pq_stat):
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )
    
    return results

def main():
    args = cli()
    args.only_output_17 = True

    json_output = []

    print('Checkpoint:', args.checkpoint)
    
    print('Max pool TH:', args.max_pool_th)
    print('Prediction Filtering Disabled: ', args.disable_pred_filter)

    # skip existing?
    if args.skip_existing:
        if os.path.exists(args.output + '.stats.json'):
            print('Output file {} exists already. Exiting.'
                  ''.format(args.output + '.stats.json'))
            return
        print('Processing: {}'.format(args.checkpoint))

    
    model_cpu, _ = network.factory_from_args(args)
    print('State of Model (Trainig)0?:', model_cpu.training)
    outputfile_name = get_output_filename(args)
    output_file = open(outputfile_name, "a")
    json_file_name = get_json_filename(args, outputfile_name)
    #####
    target_transforms = None
    # if args.oracle_data:
    target_transforms = encoder.factory(model_cpu.head_nets, model_cpu.base_net.stride, args=args)
    
    data_loader = dataloader_from_args(args, target_transforms=target_transforms, heads=model_cpu.head_nets)
    #####
    model = model_cpu.to(args.device)
    model.eval()
    # print('State of Model (Trainig)1?:', model.training)
    # if not args.disable_cuda and torch.cuda.device_count() > 1:
    #     LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
    #     model = torch.nn.DataParallel(model)
    #     model.base_net = model_cpu.base_net
    #     model.head_nets = model_cpu.head_nets
    # print('State of Model (Trainig)2?:', model.training)
    processor = decoder.factory_from_args(args, model)
    # print('State of Model (Trainig)3?:', model.training)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    coco = pycocotools.coco.COCO(args.annotation_file)
    eval_coco = EvalCoco(
        coco,
        processor,
        max_per_image=100 if args.detection_annotations else 20,
        category_ids=[] if args.detection_annotations else [1],
        iou_type='bbox' if args.detection_annotations else 'keypoints',
        considerUnvisible = not args.only_visible_keypoints,
        mask_unvisible_keypoints = args.mask_unvisible_keypoints,
        target_keypoints = args.list_mask_unvisible_keypoints
    )
    total_start = time.time()
    loop_start = time.time()
    for batch_i, (image_tensors, anns_batch, meta_batch, target_batch) in enumerate(data_loader):
        LOG.info('batch %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        if batch_i < args.skip_n:
            continue
        if args.n and batch_i >= args.n:
            break

        loop_start = time.time()
        # print('anns:', len(anns_batch))
        # print('target:', len(target_batch[0]))
        # print('anns_batch:', anns_batch)
        # print('meta_batch:', meta_batch)
        # print('target_batch:', target_batch)
        if len([a
                for anns in anns_batch
                for a in anns
                if np.any(a['keypoints'][:, 2] > 0)]) < args.min_ann:
            continue
        model.eval()
        # print('State of Model (Trainig)4?:', model.training)
        # print('state dict', model.head_nets[0].state_dict()['conv.weight'])
        # print('META:', meta_batch)
        pred_batch = processor.batch(model, image_tensors, device=args.device, oracle_masks=args.oracle_data, target_batch=target_batch)
        print('number of people in pred 1', len(pred_batch[0]))
        eval_coco.decoder_time += processor.last_decoder_time
        eval_coco.nn_time += processor.last_nn_time

        # loop over batch
        assert len(image_tensors) == len(anns_batch)
        assert len(image_tensors) == len(meta_batch)

        for image, pred, anns, meta in zip(image_tensors, pred_batch, anns_batch, meta_batch):
            eval_coco.from_predictions(pred, meta, debug=args.debug, gt=anns)

            for p in pred:
                json_data = {}
                json_data['image_id'] = meta['image_id']
                json_data['category_id'] = 1
                json_data['keypoints'] = p.json_data()['keypoints']
                json_data['score'] = p.json_data()['score']
                json_output.append(json_data)

            pq_stat = PQ.PQStat()
            if not args.disable_pan_quality:
                target_pan = target_batch[1]['panoptic']
                segments = []
                panoptic = np.zeros((image.shape[1:3]), np.uint16)
                print('--------------------------------------------------')
                print('image id', meta['image_id'])
                n_humans = 0
                print('number of people in pred', len(pred))
                for ann in pred:
                    if ann.category_id == 1 and ann.mask.any():
                        if (np.count_nonzero(ann.mask) < args.discard_smaller
                            or np.count_nonzero(ann.data[:,2]) < args.discard_lesskp
                            ):
                            continue
                        n_humans += 1
                        instance_id = 1000+n_humans
                        panoptic[ann.mask] = instance_id

                        segments.append({
                            'id': instance_id,
                            'category_id': 1,
                            
                        })
                
                print('instance pred', n_humans)

                # image_id = meta['file_name'].split('/')[-1].replace('.png', '')
                # panoptic_name = 'output/%s.png'%image_id

                background = panoptic == 0
                if background.any():
                    panoptic[background] = 2000
                    segments.append({
                        'id': 2000,
                        'category_id': 2,
                        
                    })
                print('segments pred', segments)
                # ground truth
                segments_gt = []
                panoptic_gt = target_pan.cpu().numpy()
                # unique_ids, id_cnt = np.unique(panoptic_gt, return_counts=True)
                # for u, cnt in zip(unique_ids, id_cnt):
                #     if u > 0: # and u < 3000:
                #         segments_gt.append({
                #             'id': u,
                #             'category_id': 1,
                #             'iscrowd': 0,
                #             'area': cnt,
                #         })

                for ann in anns:
                    if ann['category_id'] == 1:
                        segments_gt.append({
                            'id': ann['id'],
                            'category_id': 1,
                            'iscrowd': ann['iscrowd'],
                            'area': ann['bmask'].sum(),
                        })
                # print('unique ids ann', unique_ids)
                # print('count', id_cnt)
                print('number of people in gt', len(segments_gt))
                background = panoptic_gt == 0
                if background.any():
                    panoptic_gt[background] = 2000
                    segments_gt.append({
                        'id': 2000,
                        'category_id': 2,
                        'iscrowd': 0,
                        'area': background.sum(),
                    })
                print('segments gt', segments_gt)
                pq_stat += pq_compute_single_core(panoptic, segments, panoptic_gt, segments_gt)
                print("PQ")
                # if len(unique_ids) > 2:
                print(show_pq_results(pq_stat))

    total_time = time.time() - total_start

    # processor.instance_scorer.write_data('instance_score_data.json')

    # model stats
    count_ops = list(eval_coco.count_ops(model_cpu))
    local_checkpoint = network.local_checkpoint_path(args.checkpoint)
    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write
    write_evaluations(eval_coco, args.output, args, total_time, count_ops, file_size)

    output_file.write('\nCheckpoint: ' + str(args.checkpoint))
    output_file.write('\nMax pool TH: ' + str(args.max_pool_th))
    output_file.write('\nLeft and Right check: ' + str('Disabled' if args.disable_left_right_check else 'Enabled'))
    output_file.write('\nConsider unvisible keypoints: ' + str('No' if args.only_visible_keypoints else 'Yes'))
    output_file.write('\nMask unvisible keypoints: ' + str('Yes' if args.mask_unvisible_keypoints else 'No'))
    output_file.write('\nList of keypoints that unvisible ones are discarded'+ str(args.list_mask_unvisible_keypoints))
    output_file.write('\nDist. th knee: '+ str(args.dist_th_knee))
    output_file.write('\nDist. th ankle: '+ str(args.dist_th_ankle))
    output_file.write('\nDist. th wrist: '+ str(args.dist_th_wrist))
    output_file.write('\nDist percent: '+ str(args.dist_percent))
    output_file.write('\nOracle: ' + str(args.oracle_data))
    output_file.write('\nDecode mask first: ' + str(args.decode_masks_first))
    output_file.write('\nDecoder Filtering Strategy: ' + 'Filter Smaller than ' + str(args.decod_discard_smaller) +
                        '\tFilter with Keypoints less than ' + str(args.decod_discard_lesskp))

    if not args.disable_json_results:
        with open(json_file_name, 'w') as json_file:
            json.dump(json_output, json_file)
            print('json output results generated', json_file_name)

    stats = eval_coco.stats()
    output_file.write('\nOKS\n' + str(stats))
    output_file.write('\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = ' + str(round(stats[0], 3)))
    output_file.write('\nAverage Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = ' + str(round(stats[1], 3)))
    output_file.write('\nAverage Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = ' + str(round(stats[2], 3)))
    output_file.write('\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = ' + str(round(stats[3], 3)))
    output_file.write('\nAverage Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = ' + str(round(stats[4], 3)))
    output_file.write('\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = ' + str(round(stats[5], 3)))
    output_file.write('\nAverage Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = ' + str(round(stats[6], 3)))
    output_file.write('\nAverage Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = ' + str(round(stats[7], 3)))
    output_file.write('\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = ' + str(round(stats[8], 3)))
    output_file.write('\nAverage Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = ' + str(round(stats[9], 3)))



    if not args.disable_pan_quality:
        resss = show_pq_results(pq_stat)
        output_file.write("\n\n ---------- PQ computations ---------------------\n")
        output_file.write('\nPQ computer Filtering Strategy: ' + 'Filter Smaller than ' + str(args.discard_smaller) +
                        '\tFilter with Keypoints less than ' + str(args.discard_lesskp))
        output_file.write("\n{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}\n".format("", "PQ", "SQ", "RQ", "N"))
        output_file.write("-" * (10 + 7 * 4))
        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        for name, _isthing in metrics:
            output_file.write("\n{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * resss[name]['pq'],
                100 * resss[name]['sq'],
                100 * resss[name]['rq'],
                resss[name]['n'])
            )
            
if __name__ == '__main__':
    main()

