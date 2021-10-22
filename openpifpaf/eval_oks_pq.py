import argparse
import os
from ssl import Options
import sys
import json
import subprocess
import imageio
import numpy as np
import pickle
from pprint import pprint
import cProfile

from tqdm.auto import tqdm

from mlworkflow import PickledDataset, TransformedDataset

from dataset_utilities.court import Court
from dataset_utilities.calib import Calib, Point2D
from dataset_utilities.ds.instants_dataset import PlayerAnnotation, ViewCropperTransform, ExtractViewData

from openpifpaf.datasets.constants import COCO_KEYPOINTS
from openpifpaf.predict import main as predict

import glob
import json
import logging
import os
import cv2
import PIL
import torch
import random
from . import datasets, encoder, decoder, network, show, transforms, visualizer, __version__
from openpifpaf.datasets.collate import collate_images_targets_inst_meta_views
from openpifpaf.datasets.deepsport import DeepSportDataset, build_DeepSportBall_datasets, deepsportlab_dataset_splitter
import datetime

KiHEAD = 0.15
KiHIPS = 0.2
KiFEET = 0.2

KAPAS = {
    'head': KiHEAD,
    'hips': KiHIPS,
    'foot1': KiFEET,
    'foot2': KiFEET,
}

LOG = logging.getLogger(__name__)
import wandb

def log_wandb(in_dict):
    wandb.log(in_dict)


import panopticapi.evaluation as PQ
import copy
# import panopticapi.evaluation.PQStat as PQStat

OFFSET = 256 * 256 * 256
VOID = 0
categories_list = [{"id": 2, "name": "background", "color": [50, 50, 70], "supercategory": "void", "isthing": 0}, 
            {"id": 1, "name": "player", "color": [220, 20, 60], "supercategory": "human", "isthing": 1}, 
            {"id": 3, "name": "ball", "color": [20, 220, 60], "supercategory": "object", "isthing": 1}]

categories = {el['id']: el for el in categories_list}

class HiddenKeypointError(BaseException):
    pass

class Keypoints():
    def __init__(self, keypoints):
        self.keypoints = keypoints
    def __call__(self, name):
        idx = COCO_KEYPOINTS.index(name)
        if self.keypoints[idx*3+2] == 0:
            raise HiddenKeypointError
        return Point2D(*self.keypoints[idx*3:idx*3+2]), self.keypoints[idx*3+2]
    def all(self):
        return [self(name) for name in COCO_KEYPOINTS[0:17]]

    def swap_feet(self):
        RIGHT_ANKLE_IDX = 15
        LEFT_ANKLE_IDX = 16        
        self.keypoints[LEFT_ANKLE_IDX*3:LEFT_ANKLE_IDX*3+2], self.keypoints[RIGHT_ANKLE_IDX*3:RIGHT_ANKLE_IDX*3+2] = \
            self.keypoints[RIGHT_ANKLE_IDX*3:RIGHT_ANKLE_IDX*3+2], self.keypoints[LEFT_ANKLE_IDX*3:LEFT_ANKLE_IDX*3+2]


class PlayerSkeleton():
    def __init__(self, keypoints, category_id, score, bbox):
        n_keypoints = len(keypoints)//3
        self.keypoints = Keypoints(keypoints)
        self.category_id = category_id
        self.score = score
        self.bbox = bbox
    @property
    def head(self):
        try:
            l_ear, confidence = self.keypoints('left_ear')
        except HiddenKeypointError:
            l_ear = Point2D(np.nan, np.nan)
        try:
            r_ear, confidence = self.keypoints('right_ear')
        except HiddenKeypointError:
            r_ear = Point2D(np.nan, np.nan)
        return Point2D(np.nanmean([l_ear, r_ear], axis=0))
    @property
    def hips(self):
        try:
            l_hip, confidence = self.keypoints('left_hip')
        except HiddenKeypointError:
            l_hip = Point2D(np.nan, np.nan)
        try:
            r_hip, confidence = self.keypoints('right_hip')
        except HiddenKeypointError:
            r_hip = Point2D(np.nan, np.nan)
        return Point2D(np.nanmean([l_hip, r_hip], axis=0))
    @property
    def foot1(self):
        return self.keypoints('left_ankle')[0]
    @property
    def foot2(self):
        return self.keypoints('right_ankle')[0]

    def swap_feet(self):
        self.keypoints.swap_feet()
    @property
    def visible(self):
        for name in ["head", "hips", "foot1", "foot2"]:
            try:
                getattr(self, name)
            except HiddenKeypointError:
                continue
            return True
        else:
            return False
    def projects_in_court(self, calib: Calib, court: Court):
        MARGIN = 10
        min_y = min_x = -MARGIN
        max_x = court.w + MARGIN
        max_y = court.h + MARGIN
        for name, height in zip(["foot1", "foot2", "hips"], [-10, -10, -50]):
            try:
                point = calib.project_2D_to_3D(getattr(self, name), Z=height)
                if min_x < point.x < max_x and min_y < point.y < max_y:
                    return True
            except HiddenKeypointError:
                continue
        return False
    @property
    def confidence(self):
        acc = 0
        for idx, name in enumerate(COCO_KEYPOINTS[0:17]):
            try:
                kp, confidence = self.keypoints(name)
                acc += confidence
            except HiddenKeypointError:
                continue
        return acc/17

    @property
    def predicted_keypoints(self):
        pred_kps = []
        pred_kps_names = []
        for name in ["head", "hips", "foot1", "foot2"]:
            try:
                pr_kp = getattr(self, name)
                pred_kps.append(pr_kp)
                pred_kps_names.append(name)
            except HiddenKeypointError:
                continue
        return pred_kps, pred_kps_names
        # else:
        #     return [], []

class PlayerAnnotation2D():
    def __init__(self, annotation: PlayerAnnotation, calib: Calib):
        self.annotation = annotation
        self.calib = calib
        self.feet_swapped = False
    @property
    def head(self):
        return self.calib.project_3D_to_2D(self.annotation.head)
    @property
    def hips(self):
        return self.calib.project_3D_to_2D(self.annotation.hips)
    @property
    def foot1(self):
        if self.feet_swapped:
            return self.calib.project_3D_to_2D(self.annotation.foot2)
        return self.calib.project_3D_to_2D(self.annotation.foot1)
    @property
    def foot2(self):
        if self.feet_swapped:
            return self.calib.project_3D_to_2D(self.annotation.foot1)
        return self.calib.project_3D_to_2D(self.annotation.foot2)

    def swap_feet(self):
        self.feet_swapped = not self.feet_swapped

    def projects_in_court(self, calib: Calib, court: Court):
        MARGIN = 10
        min_y = min_x = -MARGIN
        max_x = court.w + MARGIN
        max_y = court.h + MARGIN
        for name, height in zip(["foot1", "foot2", "hips"], [0, 0, -60]):
            point = calib.project_2D_to_3D(getattr(self, name), Z=height)
            if min_x < point.x < max_x and min_y < point.y < max_y:
                return True
        return False


def compute_metrics(result_list):
    thresholds = np.linspace(.50,.95,10)
    TP = np.zeros_like(thresholds)
    FP = np.zeros_like(thresholds)
    Np = 0
    Na = 0
    for d in result_list:
        annotations = d["annotations"]
        predictions = d["predictions"]
        TP = TP + np.sum(np.array(d["oks_list"])[np.newaxis] >= thresholds[:,np.newaxis], axis=1)
        # FP = FP + np.sum(np.array(d["oks_list"])[np.newaxis] <  thresholds[:,np.newaxis], axis=1)
        Np = Np + len(predictions)
        Na = Na + len(annotations)
    return {
        "thresholds": thresholds,
        "precision": TP/Np,
        "recall": TP/Na,
        "Mprecision": np.mean(TP/Np),
        "Mrecall": np.mean(TP/Na)
    }

def OKS(a: PlayerAnnotation2D, p: PlayerSkeleton, alpha=0.8, use_dist_squared=False, enable_swap=False):
    def dist(p1, p2):
        dist_squared = np.sum((p1-p2)**2)
        return dist_squared if use_dist_squared else np.sqrt(dist_squared)
    def KS(a, p, name, kapa, s, name2=None):
        name2 = name2 or name
        try: 
            return np.exp(-dist(getattr(a, name), getattr(p, name2))/(2*s*kapa**2))
        except HiddenKeypointError:
            return np.nan

    keypoints = Point2D([a.head, a.hips, a.foot1, a.foot2])

    # scale 
    s = (np.max(keypoints.x)-np.min(keypoints.x))*(np.max(keypoints.y)-np.min(keypoints.y)) # BB area in pixels

    if np.all(a.foot1 == a.foot2): # annotation was not done on the two feet we skip OKS of feet (it only concerns a few images)
        return np.nanmean([KS(a, p, "head", KiHEAD*alpha, s), KS(a, p, "hips", KiHIPS*alpha, s)])

    pair1 = np.nanmean([KS(a, p, "head", KiHEAD*alpha, s), KS(a, p, "hips", KiHIPS*alpha, s), KS(a, p, "foot1", KiFEET*alpha, s), KS(a, p, "foot2", KiFEET*alpha, s)])
    pair2 = np.nanmean([KS(a, p, "head", KiHEAD*alpha, s), KS(a, p, "hips", KiHIPS*alpha, s), KS(a, p, "foot1", KiFEET*alpha, s, "foot2"), KS(a, p, "foot2", KiFEET*alpha, s, "foot1")])
    if enable_swap and pair2 > pair1:
        p.swap_feet()   # swap feet predictions
    return max(pair1, pair2)


class OutputInhibitor():
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        if self.name:
            print("Launching {}... ".format(self.name), end="")
        self.ps1, self.ps2 = getattr(sys, "ps1", None), getattr(sys, "ps2", None)
        if self.ps1:
            del sys.ps1
        if self.ps2:
            del sys.ps2
        self.stderr = sys.stderr
        self.fp = open(os.devnull, "w")
        sys.stderr = self.fp
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ps1:
            sys.ps1 = self.ps1
        if self.ps2:
            sys.ps2 = self.ps2
        sys.stderr = self.stderr
        self.fp.close()
        if self.name:
            print("Done.")

BALL_DIAMETER = 23

class AddBallSegmentationTargetViewFactory():
    def __call__(self, view_key, view):
        calib = view.calib
        target = np.zeros((calib.height, calib.width), dtype=np.uint8)
        for ball in [a for a in view.annotations if a.type == "ball" and calib.projects_in(a.center) and a.visible]:
            diameter = calib.compute_length2D(BALL_DIAMETER, ball.center)
            center = calib.project_3D_to_2D(ball.center)
            cv2.circle(target, center.to_int_tuple(), radius=int(diameter/2), color=1, thickness=-1)
        return {
            "mask": target
        }

class AddBallPositionFactory():
    def __call__(self, view_key, view):
        balls = [a for a in view.annotations if a.type == "ball"]
        ball = balls[0]
        if view_key.camera != ball.camera:
            return {}
        ball_2D = view.calib.project_3D_to_2D(ball.center)
        size = view.calib.compute_length2D(BALL_DIAMETER, ball.center)
        return {"x": ball_2D.x, "y": ball_2D.y, "visible": ball.visible, "size": size}


class ScaleDownFactor2Transform():
    def __call__(self, view_key, view):
        view.image = view.image[::2,::2]
        view.human_masks = view.human_masks[::2,::2]
        height, width, channels = view.image.shape
        view.calib = view.calib.scale(width, height)
        return view

class AddHumansSegmentationTargetViewFactory():
    def __call__(self, view_key, view):
        if not hasattr(view, "human_masks"):
            view.human_masks = np.zeros(view.image.shape[0:2])
        return {"human_masks": view.human_masks}

    
class AddRuleTypeFactory():
    def __call__(self, view_key, view):
        return {"rule_type": view.rule_type}

class AddAnnotationsFactory():
    def __call__(self, view_key, view):
        return {"annotations": view.annotations}

class AddCalibFactory():
    def __call__(self, view_key, view):
        return {"calib": view.calib}

def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval_oks_pq',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.cli(parser)
    decoder.cli(parser, force_complete_pose=False, instance_threshold=0.1, seed_threshold=0.5)
    show.cli(parser)
    visualizer.cli(parser)
    datasets.train_cli(parser)


    parser.add_argument("weights_file")
    
    parser.add_argument('--use-dist-squared', default=False, action='store_true', help='In OKS computation: use dist squared instead of dist')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--image-output', default=None, nargs='?', const=True,
                        help='image output file or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file or directory')
    # parser.add_argument('--batch-size', default=1, type=int,
    #                     help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='apply preprocessing to batch images')
    # parser.add_argument('--loader-workers', default=None, type=int,
    #                     help='number of workers for data loading')
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

    group.add_argument('--disable-error-detail', default=False, action='store_true')

    group.add_argument('--filter-after-matching', default=False, action='store_true')

    group.add_argument('--disable-filter-outside-pq', default=False, action='store_true')

    parser.add_argument('--oracle-masks', default=None, nargs='+',
                        help='pass centroid, semantic, and offset to use their oracle')

    
    group.add_argument('--disable-pan-quality', default=False, action='store_true')

    parser.add_argument('--discard-smaller', default=0, type=int,
                        help='discard smaller than')

    parser.add_argument('--discard-lesskp', default=0, type=int,
                        help='discard with number of keypoints less than')

    parser.add_argument('--matching-oks-th', default=0.0, type=float,
                        help='discard matching with oks less than')

                        
    parser.add_argument('--matching-reference', default='annotations', 
                        choices=['annotations', 'predictions'],
                        help='choose annotations or predictions')

    group.add_argument('--disable-oks', default=False, action='store_true')

    group.add_argument('--disable-wandb', default=False, action='store_true')
                    
    group.add_argument('--wandb-dir', default='wandb/',
                    help='wandb directory')

    group.add_argument('--use-crops', default=False, action='store_true')
    # group.add_argument("--pickled-dataset", required=True)
    # group.add_argument('--focus-object', default=None)

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

def build_DeepSport_test_dataset(pickled_dataset_filename, validation_set_size_pc, square_edge, target_transforms, preprocess, focus_object=None, config=None, dataset_fold=None, use_crops=False):
    dataset = PickledDataset(pickled_dataset_filename)
    keys = list(dataset.keys.all())
    
    if dataset_fold != "all":
        keys = deepsportlab_dataset_splitter(keys, dataset_fold, 0, validation_set_size_pc)["testing"]

    if use_crops:
        transforms = [
        ViewCropperTransform(output_shape=(square_edge,square_edge), def_min=30, def_max=80, max_angle=8, focus_object=focus_object),
        ExtractViewData(
            AddBallPositionFactory(),
            AddBallSegmentationTargetViewFactory(),
            AddHumansSegmentationTargetViewFactory(),
            AddRuleTypeFactory(),
            AddAnnotationsFactory(),
            AddCalibFactory(),
        )
        ]
    else:
        transforms = [
            ScaleDownFactor2Transform(),
            ExtractViewData(
                AddBallPositionFactory(),
                AddBallSegmentationTargetViewFactory(),
                AddHumansSegmentationTargetViewFactory(),
                AddRuleTypeFactory(),
                AddAnnotationsFactory(),
                AddCalibFactory(),
            )
        ]
    dataset = TransformedDataset(dataset, transforms)
    return DeepSportDataset(dataset, keys, target_transforms, preprocess, config, oks_computation=True)

def get_output_filename(args):
    import os
    if not os.path.exists('oks_logs'):
        os.makedirs('oks_logs')
    now = datetime.datetime.now().strftime('%y%m%d-%H%M%S.%f')
    out = 'oks_logs/{}-{}'.format(now, args.checkpoint.split('/')[-1])

    return out + '.log'


def main():
    args = cli()
   
    args.checkpoint = args.weights_file
    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)
    if model.head_nets[0].meta.name == 'pan':
        pan_id = 0
    elif model.head_nets[1].meta.name == 'pan':
        pan_id = 1
    else:
        pan_id = 1

    output_file = open(get_output_filename(args), "a")
    

    target_transforms = encoder.factory(model.head_nets, model.base_net.stride)
    heads = []
    for hd in model.head_nets:
        heads.append(hd.meta.name)

    dataset = build_DeepSport_test_dataset(
        pickled_dataset_filename=args.deepsport_pickled_dataset,
        validation_set_size_pc=15, square_edge=args.square_edge, target_transforms=target_transforms,
        preprocess=preprocess, focus_object=args.focus_object, config=heads, dataset_fold=args.dataset_fold, use_crops=args.use_crops)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,

        collate_fn=collate_images_targets_inst_meta_views,)

    result_list = []

    images_with_wrong_feet = 0
    error_detail_dict = {}
    sum_all_cases_true = 0

    for error_type in ['Good', 'Jitter', 'Inversion', 'Swap', 'Miss', 'Nan']:
        error_detail_dict[error_type] = {}
        for name in ['head','hips','foot1','foot2']:
            error_detail_dict[error_type][name] = 0
    Background_FP = 0
    Background_FN = 0
    Total_people_pred = 0
    Total_people_annot = 0
    swap_ids = []

    pq_stat = PQ.PQStat()
    
    
    for batch_i, (image_tensors_batch, target_batch, meta_batch, views_batch, keys_batch) in enumerate(tqdm(data_loader)):
        


        pred_batch = processor.batch(model, image_tensors_batch, device=args.device, oracle_masks=args.oracle_masks, target_batch=target_batch) #, numb=batch_i)
        
        # unbatch
        assert len(pred_batch)==len(views_batch)

        for image, pred, meta, view, key, target_pan in zip(image_tensors_batch, pred_batch, meta_batch, views_batch, keys_batch, target_batch[pan_id]['panoptic']):

            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            

            if not args.disable_oks and args.matching_reference == 'annotations':
                court = Court(view['rule_type'])

                predictions_dict = {}
                for i, p in enumerate(pred):
                    p_a = PlayerSkeleton(**p.json_data())
                    predictions_dict[i] = {
                        'p_a': p_a,
                        'id_matched': -1,
                        'inside_the_court': True,
                        'oks_matched': 0,
                        'pred': p,
                    }
                    for kp_name in ['head','hips','foot1','foot2']:
                        try:
                            kp = getattr(p_a, kp_name)
                            predictions_dict[i][kp_name] = kp
                        except:
                            predictions_dict[i][kp_name] = 'HiddenKeypointError'
                
                
                #  annotations = [PlayerAnnotation2D(a, view['calib']) for a in view['annotations'] if a.type == "player" and a.camera == key.camera and all([view['calib'].projects_in(kp) for kp in [a.head, a.hips, a.foot1, a.foot2]])]
                
                annotations_dict = {}
                for i, a in enumerate(view['annotations']):
                    # if a.type == "player" and a.camera == key.camera and all([view['calib'].projects_in(kp) for kp in [a.head, a.hips, a.foot1, a.foot2]]):  
                    if a.type == "player" and all([view['calib'].projects_in(kp) for kp in [a.head, a.hips, a.foot1, a.foot2]]):  
                        p_a = PlayerAnnotation2D(a, view['calib'])
                        annotations_dict[i] = {
                            'p_a': p_a,
                            'id_matched': -1,
                            'inside_the_court': True,
                            'oks_matched': 0,
                        }
                        for kp_name in ['head','hips','foot1','foot2']:
                            try:
                                kp = getattr(p_a, kp_name)
                                annotations_dict[i][kp_name] = kp
                            except:
                                annotations_dict[i][kp_name] = 'HiddenKeypointError'
                                
                
                
                
                if any([np.all(a['p_a'].foot1 == a['p_a'].foot2) for _,a in annotations_dict.items()]):
                    logging.warning(f"'{key}' has un-corrected feet annotation. We will skip the feet in OKS computation for this item")
                    images_with_wrong_feet += 1

                matching = {}
                oks_list = []
                matching_oks = {}
                if annotations_dict:
                    for a_id,a_k in annotations_dict.items():
                        a = a_k['p_a']
                        if not predictions_dict or all([p_k['id_matched'] > -1 for _,p_k in predictions_dict.items()]):
                            break
                        try:
                            idx = np.nanargmax([OKS(a, p_k['p_a'], use_dist_squared=args.use_dist_squared) if p_k['id_matched'] == -1 else 0 for _,p_k in predictions_dict.items()])
                        except ValueError:
                            print('all of predictions are NaN')
                            continue
                        oks_max = round(np.nanmax([OKS(a, p_k['p_a'], use_dist_squared=args.use_dist_squared) if p_k['id_matched'] == -1 else 0 for _,p_k in predictions_dict.items()]), 2)
                        if oks_max < args.matching_oks_th and not np.isnan(oks_max):
                            continue
                        predictions_dict[idx]['id_matched'] = a_id
                        predictions_dict[idx]['oks_matched'] = oks_max
                        a_k['id_matched'] = idx
                        a_k['oks_matched'] = oks_max
                        a = a_k['p_a']
                        
                        matching[predictions_dict[idx]['p_a']] = a_k['p_a']
                        matching_oks[predictions_dict[idx]['p_a']] = oks_max
                        oks_list.append(OKS(a, predictions_dict[idx]['p_a'], alpha=0.8, enable_swap=True))

                for _, p_k in predictions_dict.items():
                    p = p_k['p_a']
                    if (p.projects_in_court(view['calib'], court) and p.visible) or p_k['id_matched'] > -1:
                        continue
                    else:
                        p_k['inside_the_court'] = False
                        
                
                result_list.append({
                    "predictions": [p_k['p_a'] for _, p_k in predictions_dict.items() if p_k['inside_the_court']],
                    "annotations": [a_k['p_a'] for _, a_k in annotations_dict.items() if a_k['inside_the_court']],
                    "matching": matching,
                    "oks_list": oks_list,
                })

                Total_people_annot += len([a_k['p_a'] for a_k in annotations_dict.values() if a_k['inside_the_court']])      # to have count of people in annotations
                Background_FN += len([a_k['p_a'] for _, a_k in annotations_dict.items() if a_k['id_matched']==-1 and a_k['inside_the_court']])       # to have count of people after matching
                

                if not args.disable_error_detail:
                    for p_id, p_k in predictions_dict.items():
                        if not p_k['inside_the_court']:
                            continue
                        
                        if p_k['id_matched'] == -1:
                            Background_FP += 1
                            continue
                        else:
                            match = annotations_dict[p_k['id_matched']]['p_a']
                        p = p_k['p_a']

                        for idddx, (kp, kp_name) in enumerate(zip(*p.predicted_keypoints)):
                            km = getattr(match, kp_name)
                            ks_result = KS(match, kp, kp_name, use_dist_squared=args.use_dist_squared)
                            if np.isnan(ks_result):
                                
                                error_detail_dict['Nan'][kp_name] += 1
                                
                                continue
                            if ks_result >= 0.85:
                                
                                error_detail_dict['Good'][kp_name] += 1
                                
                            elif 0.5 <= ks_result < 0.85:
                                
                                error_detail_dict['Jitter'][kp_name] += 1
                                
                            else:
                                for_done = False
                                for a_id, a_k in annotations_dict.items():
                                    a = a_k['p_a']
                                    if a == match:
                                        continue
                                    for ka_name in ['head','hips','foot1','foot2']:
                                        for_done = False
                                        ks_result = KS(a, kp, ka_name, use_dist_squared=args.use_dist_squared)
                                        if np.isnan(ks_result):
                                            continue
                                        elif ks_result >= 0.5:
                                            error_detail_dict['Swap'][kp_name] += 1
                                            swap_ids.append(batch_i)
                                            for_done = True
                                            break
                                    if for_done:
                                        break
                                if for_done:
                                    continue

                                error_detail_dict['Miss'][kp_name] += 1
                        sum_all_cases = sum([sum(edd.values()) for edd in error_detail_dict.values()])
                        sum_all_cases_true += len(p.predicted_keypoints[0]) 
                        assert sum_all_cases_true == sum_all_cases, str(sum_all_cases_true)+ "\n"+ str(sum_all_cases)+'\nid: '+ str(batch_i)

                    Total_people_pred += len([p for p in predictions_dict.values() if p['inside_the_court']])

            elif not args.disable_oks and args.matching_reference == 'predictions':
                court = Court(view['rule_type'])

                predictions = [PlayerSkeleton(**ann.json_data()) for ann in pred]
                
                annotations = [PlayerAnnotation2D(a, view['calib']) for a in view['annotations'] if a.type == "player" and a.camera == key.camera and all([view['calib'].projects_in(kp) for kp in [a.head, a.hips, a.foot1, a.foot2]])]


                if not args.filter_after_matching:
                    predictions = [p for p in predictions if p.projects_in_court(view['calib'], court) and p.visible]
                    # remove remaining annotations that lie outside the court
                    annotations = [a for a in annotations if a.projects_in_court(view['calib'], court)]
                    Total_people_annot += len(annotations)      # to have count of people in annotations

                if any([np.all(a.foot1 == a.foot2) for a in annotations]):
                    logging.warning(f"'{key}' has un-corrected feet annotation. We will skip the feet in OKS computation for this item")
                    images_with_wrong_feet += 1

                matching = {}
                oks_list = []
                

                
                if predictions:
                    for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
                        if not annotations:
                            break
                        idx = np.argmax([OKS(a, p, use_dist_squared=args.use_dist_squared) for a in annotations])
                        oks_max = round(np.max([OKS(a, p, use_dist_squared=args.use_dist_squared) for a in annotations]), 2)
                        if oks_max < args.matching_oks_th:
                            continue
                        matching[p] = annotations[idx]
                        oks_list.append(OKS(annotations[idx], p, alpha=0.8, enable_swap=True))
                        del annotations[idx]


                result_list.append({
                    "predictions": predictions,
                    "annotations": annotations + list(matching.values()),
                    "matching": matching,
                    "oks_list": oks_list,
                })

                Background_FN += len(annotations)       # to have count of people after matching
                
                
                if not args.disable_error_detail:

                    for p_id, p in enumerate(predictions):
                        try:
                            match = matching[p]
                        except KeyError:

                            Background_FP += 1
                            continue

                        for idddx, (kp, kp_name) in enumerate(zip(*p.predicted_keypoints)):
                            km = getattr(match, kp_name)
                            ks_result = KS(match, kp, kp_name, use_dist_squared=args.use_dist_squared)
                            if np.isnan(ks_result):
                                
                                error_detail_dict['Nan'][kp_name] += 1
                                
                                continue
                            if ks_result >= 0.85:
                                
                                error_detail_dict['Good'][kp_name] += 1
                            elif 0.5 <= ks_result < 0.85:
                                error_detail_dict['Jitter'][kp_name] += 1
                            else:
                                for ka_name in list(set(['head','hips','foot1','foot2']) - set([kp_name])):
                                    
                                    for_done = False
                                    ks_result = KS(match, kp, ka_name, use_dist_squared=args.use_dist_squared)
                                    if np.isnan(ks_result):
                                        continue
                                    elif ks_result >= 0.5:
                                        error_detail_dict['Inversion'][kp_name] += 1
                                        
                                        for_done = True
                                        break

                                if for_done:
                                    continue

                                for a in list(set(annotations + list(matching.values())) -set([match])):
                                    for ka_name in ['head','hips','foot1','foot2']:
                                        for_done = False
                                        ks_result = KS(a, kp, ka_name, use_dist_squared=args.use_dist_squared)
                                        if np.isnan(ks_result):
                                            continue
                                        elif ks_result >= 0.5:
                                            error_detail_dict['Swap'][kp_name] += 1
                                            for_done = True
                                            break
                                    if for_done:
                                        break
                                if for_done:
                                    continue

                                error_detail_dict['Miss'][kp_name] += 1
                        sum_all_cases = sum([sum(edd.values()) for edd in error_detail_dict.values()])
                        sum_all_cases_true += len(p.predicted_keypoints[0]) 

                        assert sum_all_cases_true == sum_all_cases, str(sum_all_cases_true)+ "\n"+ str(sum_all_cases)
                
                    Total_people_pred += len(predictions)

            
            if not args.disable_pan_quality:
                segments = []
                panoptic = np.zeros((image.shape[1:3]), np.uint16)

                n_humans = 0
                
                if args.matching_reference == 'annotations' and not args.disable_filter_outside_pq:
                    for _, p_k in predictions_dict.items():
                        if p_k['inside_the_court']:
                            ann = p_k['pred']
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
                else:
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
                

                background = panoptic == 0
                if background.any():
                    panoptic[background] = 2000
                    segments.append({
                        'id': 2000,
                        'category_id': 2,
                        
                    })
                
                # ground truth
                segments_gt = []
                panoptic_gt = copy.deepcopy(target_pan.cpu().numpy())
                unique_ids, id_cnt = np.unique(panoptic_gt, return_counts=True)
                for u, cnt in zip(unique_ids, id_cnt):
                    if u > 0 and u < 3000:
                        segments_gt.append({
                            'id': u,
                            'category_id': 1,
                            'iscrowd': 0,
                            'area': cnt,
                        })

                background = panoptic_gt == 0
                if background.any():
                    panoptic_gt[background] = 2000
                    segments_gt.append({
                        'id': 2000,
                        'category_id': 2,
                        'iscrowd': 0,
                        'area': background.sum(),
                    })
                
                pq_stat += pq_compute_single_core(panoptic, segments, panoptic_gt, segments_gt)
                if len(unique_ids) > 2:
                    print(show_pq_results(pq_stat))


        if batch_i%10 == 0:
            pprint(compute_metrics(result_list))
            if not args.disable_pan_quality:
                print(show_pq_results(pq_stat))

    logging.warning(f"Images with wrong feet annotations: {images_with_wrong_feet}")
    pickle.dump(result_list, open(f"{args.weights_file}_OKS_tmp_results.pickle", "wb"))
    pprint(compute_metrics(result_list))
    print('Checkpoint: ', args.weights_file)
    print('Oracle: ', args.oracle_masks)
    print('Kapas: ', KAPAS)
    print('Filter the prediction and annotations after the matching: ', args.filter_after_matching)

    output_file.write('\nCheckpoint: ' + str(args.weights_file))
    output_file.write('\nMatching Reference: '+ str(args.matching_reference))
    output_file.write('\nOracle: ' + str(args.oracle_masks))
    output_file.write('\nKapas: ' + str(KAPAS))
    output_file.write('\nMatching OKS Threshold: ' + str(args.matching_oks_th))
    output_file.write('\nDecode mask first: ' + str(args.decode_masks_first))
    output_file.write('\nFilter the prediction and annotations after the matching:: ' + str(args.filter_after_matching))
    output_file.write('\nFilter outside the court preds for PQ: '+ str('Disabled' if args.disable_filter_outside_pq else 'Enabled'))
    output_file.write('\nPrediction Filter: ' + str('Disabled' if args.disable_pred_filter else 'Enabled'))
    output_file.write('\nDecoder Filtering Strategy: ' + 'Filter Smaller than ' + str(args.decod_discard_smaller) +
                        '\tFilter with Keypoints less than ' + str(args.decod_discard_lesskp))

    output_file.write('\nUsing PanopticDeepLab for Decoding:' + str(args.use_panoptic_deeplab_output_decode))
    output_file.write('\nUse crops: ' + str(args.use_crops))
    
    wandb_result_dict = {}

    if not args.disable_oks:
        resss = compute_metrics(result_list)
        output_file.write('\nOKS: ' + str())
        for k,v in resss.items():
            output_file.write('\n\t' + str(k) + ' : ' + str(v))
        output_file.write('\nCheckpoint: ' + str(args.weights_file))
        sizes = [sum(l.values()) for l in error_detail_dict.values()]
        output_file.write('\nError Detail per keypoint: ' + str(error_detail_dict))
        output_file.write('\nError Detail: ' + str(sizes))
        output_file.write('\nError Detail Percentage: ' + str([round((s/sum(sizes)*100),1) for s in sizes]))
        output_file.write('\nBackgound FP: ' + str(Background_FP))
        output_file.write('\nTotal People pred: ' + str(Total_people_pred))
        if Total_people_pred > 0:
            output_file.write('\nBackgound FP rate: ' + str(round((Background_FP/Total_people_pred)*100, 3)))
        output_file.write('\nBackgound FP: ' + str(Background_FP))

        output_file.write('\nBackgound FN: ' + str(Background_FN))
        output_file.write('\nTotal People annot: ' + str(Total_people_annot))
        if Total_people_annot > 0:
            output_file.write('\nBackgound FN rate: ' + str(round((Background_FN/Total_people_annot)*100, 3)))
        output_file.write('\nF1 Score: ' + str(round((resss['Mprecision'] * resss['Mrecall'])/(resss['Mprecision'] + resss['Mrecall']), 3)))

        wandb_result_dict = {
            'AP': resss['Mprecision'],
            'AR': resss['Mrecall'],
            'Backgound FP rate': round((Background_FP/Total_people_pred)*100, 3) if Total_people_pred > 0 else -1,
            'Backgound FN rate': round((Background_FN/Total_people_annot)*100, 3) if Total_people_annot > 0 else -1,
        }
        for k, v in error_detail_dict.items():
            wandb_result_dict[k] = v

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

    output_file.close()


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


def plot_pie_chart(error_detail, Background_FP, Total_people_pred, Background_FN, Total_people_annot):
    import matplotlib.pyplot as plt
    fig1, axes = plt.subplots(1,6, figsize=(50,5))
    #add colors
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    def func(pct, sizes):
        absolute = int(round(pct/100.*np.sum(sizes)))
        if pct >= 5:
            return "{:.1f}".format(pct)
        else:
            return ''
        
        
    for idx, ax in zip(error_detail.keys(),axes):
        
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = [l for l in error_detail[idx].keys()]
        sizes = [l for l in error_detail[idx].values()]
        explode = []
        for s in sizes:
            if s/sum(sizes) > .05:
                explode.append(0)
            else:
                explode.append(0.05)
        wedges, texts, data = ax.pie(sizes, colors=colors, explode=explode, autopct=lambda pct: func(pct,sizes),
                shadow=False, startangle=90)
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title(idx)

        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
        
        for i, p in enumerate(wedges):
            p.set_edgecolor('black')
            if 0<sizes[i]/sum(sizes)<0.05:
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        #         connectionstyle = "angle,angleA=0,angleB=90"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(round((sizes[i]/sum(sizes)) *100,1), xy=(x, y), xytext=(1.25*np.sign(x), 1.4*y),
                            horizontalalignment=horizontalalignment, **kw)
                
    axes[0].legend(wedges, labels,
            title="Detection Type",
            loc="center left",
            ncol=4,
            bbox_to_anchor=(0, 0, 0, -.5))
    # plt.savefig('image/pie_chart_detail.png')


    explode = (0, 0.1, 0, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax = plt.subplots(1,1, figsize=(10,10))

    # for idx, ax in zip(error_detail['good'].keys(),axes):
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    colors = ['#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC', '#DDDDDD']
    labels = [l for l in error_detail.keys()]
    sizes = [sum(l.values()) for l in error_detail.values()]
    explode = []
    for s in sizes:
        if s/sum(sizes) > .05:
            explode.append(0)
        else:
            explode.append(0.05)

    def func(pct, sizes):
        absolute = int(round(pct/100.*np.sum(sizes)))
        if pct >= 5:
            return "{:.1f}".format(pct)
        else:
            return ''
        


    wedges, texts, data = ax.pie(sizes, explode=explode, colors=colors, autopct=lambda pct: func(pct,sizes) ,
            shadow=False, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # ax.legend(loc='lower left',ncol=6)
    ax.set_title('Total')
    ax.legend(wedges, labels,
            title="Detection Type",
            loc="center left",
            ncol=6,
            bbox_to_anchor=(1, 0, 0.5, 1))

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        p.set_edgecolor('black')
        if sizes[i]/sum(sizes)<0.05:
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    #         connectionstyle = "angle,angleA=0,angleB=90"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(round((sizes[i]/sum(sizes)) *100,1), xy=(x, y), xytext=(1.25*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)
    # plt.savefig('image/pie_chart_total.png')

    print('Error Detail = ', [round((s/sum(sizes)*100),1) for s in sizes])

    print('Backgound FP:', Background_FP)
    print('Total People pred:', Total_people_pred)
    if Total_people_pred > 0:
        print('Backgound FP rate:', round((Background_FP/Total_people_pred)*100, 3))
    

    print('Backgound FN:', Background_FN)
    print('Total People annot:', Total_people_annot)
    if Total_people_annot > 0:
        print('Backgound FN rate:', round((Background_FN/Total_people_annot)*100, 3))

def KS(a, p, name, use_dist_squared=False):
    def dist(p1, p2):
        dist_squared = np.sum((p1-p2)**2)
        return dist_squared if use_dist_squared else np.sqrt(dist_squared)
    keypoints = Point2D([a.head, a.hips, a.foot1, a.foot2])
    # scale 
    s = (np.max(keypoints.x)-np.min(keypoints.x))*(np.max(keypoints.y)-np.min(keypoints.y)) # BB area in pixels

    try:
        a = getattr(a, name)
    except HiddenKeypointError:
        return np.nan
    kapa = KAPAS[name]
    return np.exp(-dist(a, p)/(2*s*kapa**2))
    

if __name__ == "__main__":
    main()
