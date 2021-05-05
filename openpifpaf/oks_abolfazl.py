import argparse
import os
import sys
import json
import subprocess
import imageio
import numpy as np
import pickle
from pprint import pprint


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
                print('HIDDEN KEYPOINT ERROR!!!!!!!!!!!!!!!!!!!')
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
        # print('\n about to swap:')
        # print('pair1 and pair2', pair1, pair2)
        # print('ann', a)
        # print('pred', p)
        # try:
        #     print('Foot1',p.foot1)
        # except HiddenKeypointError:
        #     print('Foot1 hidden')
        # try:
        #     print('Foot2',p.foot2)
        # except HiddenKeypointError:
        #     print('Foot2 hidden')
        p.swap_feet()   # swap feet predictions
        # print('swapped:')
        # print('ann', a)
        # print('pred', p)
        # try:
        #     print('Foot1',p.foot1)
        # except HiddenKeypointError:
        #     print('Foot1 hidden')
        # try:
        #     print('Foot2',p.foot2)
        # except HiddenKeypointError:
        #     print('Foot2 hidden')

        # print('\n about to swap:')
        # print('pair1 and pair2', pair1, pair2)
        # print('ann', a)
        # print('pred', p)
        # print(getattr(a, 'foot1'))
        # print(getattr(a, 'foot2'))
        # print(a.feet_swapped)
        # a.swap_feet()   # swap feet predictions
        # print(a.feet_swapped)
        # print('swapped:')
        # print('ann', a)
        # print('pred', p)
        # print(getattr(a, 'foot1'))
        # print(getattr(a, 'foot2'))
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
        prog='python3 -m openpifpaf.oks_abolfazl',
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
    # parser.add_argument('images', nargs='*',
    #                     help='input images')

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

    parser.add_argument('--oracle-masks', default=None, nargs='+',
                        help='pass centroid, semantic, and offset to use their oracle')

    
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

    # glob
    # if args.glob:
    #     args.images += glob.glob(args.glob)
    # if not args.images:
    #     raise Exception("no image files given")

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

def build_DeepSport_test_dataset(pickled_dataset_filename, validation_set_size_pc, square_edge, target_transforms, preprocess, focus_object=None, config=None, dataset_fold=None):
    dataset = PickledDataset(pickled_dataset_filename)
    keys = list(dataset.keys.all())
    
    if dataset_fold != "all":
        keys = deepsportlab_dataset_splitter(keys, dataset_fold, 0, validation_set_size_pc)["testing"]

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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("weights_file")
    # args = parser.parse_args()
    args = cli()
    args.checkpoint = args.weights_file

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    output_file = open(get_output_filename(args), "a")
    
    #assert args.dataset_fold == "DeepSport", f"You should not eval OKS on another split than DeepSport. Current slit is {args.dataset_fold}"

    target_transforms = encoder.factory(model.head_nets, model.base_net.stride)
    heads = []
    for hd in model.head_nets:
        heads.append(hd.meta.name)

    dataset = build_DeepSport_test_dataset(
        pickled_dataset_filename=args.deepsport_pickled_dataset,
        validation_set_size_pc=15, square_edge=args.square_edge, target_transforms=target_transforms,
        preprocess=preprocess, focus_object=args.focus_object, config=heads, dataset_fold=args.dataset_fold)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        # worker_init_fn=reset_pickled_datasets,
        collate_fn=collate_images_targets_inst_meta_views,)

    result_list = []

    images_with_wrong_feet = 0
    # logging.warning("Abolfazl, you didn't implement the oracle selection yet !!!!!!")
    error_detail_dict = {}
    sum_all_cases_true = 0
    # nan_counter = 0
    for error_type in ['Good', 'Jitter', 'Inversion', 'Swap', 'Miss', 'Nan']:
        error_detail_dict[error_type] = {}
        for name in ['head','hips','foot1','foot2']:
            error_detail_dict[error_type][name] = 0
    Background_FP = 0
    Background_FN = 0
    Total_people_pred = 0
    Total_people_annot = 0

    for batch_i, (image_tensors_batch, target_batch, meta_batch, views_batch, keys_batch) in enumerate(tqdm(data_loader)):

        pred_batch = processor.batch(model, image_tensors_batch, device=args.device, oracle_masks=args.oracle_masks, target_batch=target_batch)
        
        # unbatch
        assert len(pred_batch)==len(views_batch)


        for pred, meta, view, key in zip(pred_batch, meta_batch, views_batch, keys_batch):
            # print('-------------------------new Image----------------------------')
            # print('Number of people in this image',len(pred))

            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            # print(view.keys())
            court = Court(view['rule_type'])
            # print('Pred:', len([ann.mask for ann in pred]))
            # raise
            
            
            predictions = [PlayerSkeleton(**ann.json_data()) for ann in pred]
            
            annotations = [PlayerAnnotation2D(a, view['calib']) for a in view['annotations'] if a.type == "player" and a.camera == key.camera and all([view['calib'].projects_in(kp) for kp in [a.head, a.hips, a.foot1, a.foot2]])]
            # print('Predictions', len(predictions))
            # print('Annotations', len(annotations))

            if not args.filter_after_matching:
                predictions = [p for p in predictions if p.projects_in_court(view['calib'], court) and p.visible]
                # remove remaining annotations that lie outside the court
                annotations = [a for a in annotations if a.projects_in_court(view['calib'], court)]
                # print('Number of people in this image after deleting outside of the court',len(predictions))
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
                    matching[p] = annotations[idx]
                    oks_list.append(OKS(annotations[idx], p, alpha=0.8, enable_swap=True))
                    del annotations[idx]

            
            # if args.filter_after_matching:
            #     predictions = [p for p in predictions if p.projects_in_court(view['calib'], court) and p.visible]
            #     # remove remaining annotations that lie outside the court
            #     annotations = [a for a in annotations if a.projects_in_court(view['calib'], court)]
            #     print('\n 22222222222222222')
            #     print('\n predictions', predictions)
            #     print('\n annotations', annotations)
            #     print('\n matching', matching)
            #     matching_new = {}
            #     for ixx,(p,a) in enumerate(matching.items()):
            #         if p.projects_in_court(view['calib'], court) and p.visible and a.projects_in_court(view['calib'], court):
            #             matching_new[p] = a
            #     matching = matching_new
            #     print('matching_new', matching_new)
            #     print('matching', matching)
            #     # raise
            #     print('Number of people in this image after deleting outside of the court',len(predictions))
            #     Total_people_annot += len(annotations)      # to have count of people in annotations
            #     Total_people_annot += len(matching)
            

            # with open(f"oks_{batch_i}.pickle", "wb") as f:
            #     pickle.dump(key, f)
            #     pickle.dump(view, f)
            #     pickle.dump(predictions, f)
            #     pickle.dump(annotations, f)
            #     pickle.dump(matching, f)
            #     pickle.dump(oks_list, f)
            # raise
            result_list.append({
                "predictions": predictions,
                "annotations": annotations + list(matching.values()),
                "matching": matching,
                "oks_list": oks_list,
            })

            Background_FN += len(annotations)       # to have count of people after matching
            
            
            if not args.disable_error_detail:
                # print('number of people after OKS computations', len(predictions))
                for p_id, p in enumerate(predictions):
                    try:
                        match = matching[p]
                    except KeyError:
                        # print('Matching missing', p_id)
                        Background_FP += 1
                        continue
                    # print('~~~~~ person # ', p_id)
                    # print('prediction:', p.predicted_keypoints)

                    for idddx, (kp, kp_name) in enumerate(zip(*p.predicted_keypoints)):
                        km = getattr(match, kp_name)
                        # print('start for ', idddx, kp_name)
                        ks_result = KS(match, kp, kp_name, use_dist_squared=args.use_dist_squared)
                        if np.isnan(ks_result):
                            # nan_counter += 1
                            # nan_count[kp_name] += 1
                            error_detail_dict['Nan'][kp_name] += 1
                            # print('idx nan:', idddx)
                            continue
                        if ks_result >= 0.85:
                            # print('idx good:', idddx)
                            error_detail_dict['Good'][kp_name] += 1
                            # good[kp_name] += 1
                        elif 0.5 <= ks_result < 0.85:
                            # print('idx jitter:', idddx)
                            error_detail_dict['Jitter'][kp_name] += 1
                            # jitter[kp_name] += 1
                        else:
                            for ka_name in list(set(['head','hips','foot1','foot2']) - set([kp_name])):
                                # if ka_name == kp_name:
                                    # continue
                                for_done = False
                                ks_result = KS(match, kp, ka_name, use_dist_squared=args.use_dist_squared)
                                if np.isnan(ks_result):
                                    continue
                                elif ks_result >= 0.5:
                                    # print('idx inversion:', idddx)
                                    error_detail_dict['Inversion'][kp_name] += 1
                                    # inversion[kp_name] += 1
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
                                        # print('idx swap:', idddx)
                                        error_detail_dict['Swap'][kp_name] += 1
                                        # swap[kp_name] += 1
                                        for_done = True
                                        break
                                if for_done:
                                    break
                            if for_done:
                                continue

                            # print('idx miss:', idddx)
                            error_detail_dict['Miss'][kp_name] += 1
                            # miss[kp_name] += 1

                    # sum_all_cases = sum(good.values())+sum(jitter.values())+sum(inversion.values())+sum(swap.values())+sum(miss.values())+sum(nan_count.values())
                    sum_all_cases = sum([sum(edd.values()) for edd in error_detail_dict.values()])
                    sum_all_cases_true += len(p.predicted_keypoints[0]) 
                    # print('Error detail:', error_detail_dict)
                    # print('Backgound FP:', Background_FP)
                    # print('Total People pred:', Total_people_pred)
                    # print('Backgound FN:', Background_FN)
                    # print('Total People annot:', Total_people_annot)
                    # if Total_people_pred > 0:
                    #     print('Backgound FP rate:', round((Background_FP/Total_people_pred)*100, 3))
                    # if Total_people_annot > 0:
                    #     print('Backgound FN rate:', round((Background_FN/Total_people_annot)*100, 3))
                    assert sum_all_cases_true == sum_all_cases, str(sum_all_cases_true)+ "\n"+ str(sum_all_cases)
            
                Total_people_pred += len(predictions)

        if batch_i%10 == 0:
            pprint(compute_metrics(result_list))
            # if not args.disable_error_detail:
            #     plot_pie_chart(error_detail_dict, Background_FP, Total_people_pred, Background_FN, Total_people_pred)

    logging.warning(f"Images with wrong feet annotations: {images_with_wrong_feet}")
    pickle.dump(result_list, open(f"{args.weights_file}_OKS_tmp_results.pickle", "wb"))
    pprint(compute_metrics(result_list))
    print('Checkpoint: ', args.weights_file)
    print('Oracle: ', args.oracle_masks)
    print('Kapas: ', KAPAS)
    print('Filter the prediction and annotations after the matching: ', args.filter_after_matching)

    output_file.write('\nCheckpoint: ' + str(args.weights_file))
    output_file.write('\nOracle: ' + str(args.oracle_masks))
    output_file.write('\nKapas: ' + str(KAPAS))
    output_file.write('\nFilter the prediction and annotations after the matching:: ' + str(args.filter_after_matching))
    resss = compute_metrics(result_list)
    output_file.write('\nOKS: ' + str())
    for k,v in resss.items():
        output_file.write('\n\t' + str(k) + ' : ' + str(v))
    output_file.write('\nCheckpoint: ' + str(args.weights_file))
    sizes = [sum(l.values()) for l in error_detail_dict.values()]
    output_file.write('\nError Detail: ' + str([round((s/sum(sizes)*100),1) for s in sizes]))
    output_file.write('\nBackgound FP: ' + str(Background_FP))
    output_file.write('\nTotal People pred: ' + str(Total_people_pred))
    if Total_people_pred > 0:
        output_file.write('\nBackgound FP rate: ' + str(round((Background_FP/Total_people_pred)*100, 3)))
    output_file.write('\nBackgound FP: ' + str(Background_FP))

    output_file.write('\nBackgound FN: ' + str(Background_FN))
    output_file.write('\nTotal People annot: ' + str(Total_people_annot))
    if Total_people_annot > 0:
        output_file.write('\nBackgound FN rate: ' + str(round((Background_FN/Total_people_annot)*100, 3)))

    output_file.close()


    if not args.disable_error_detail:
        plot_pie_chart(error_detail_dict, Background_FP, Total_people_pred, Background_FN, Total_people_annot)

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

    # if name in ['hips', 'head']:
    try:
        a = getattr(a, name)
    except HiddenKeypointError:
        return np.nan
    kapa = KAPAS[name]
    # try: 
    return np.exp(-dist(a, p)/(2*s*kapa**2))
    # except HiddenKeypointError:
    #     return np.nan

    # else:
    #     flag1 = False
    #     flag2 = False
    #     kapa = KAPAS[name]
    #     try:
    #         a1 = getattr(a, 'foot1')
    #         ks1 = np.exp(-dist(a1, p)/(2*s*kapa**2))
    #     except HiddenKeypointError:
    #         flag1 = True
    #         ks1 = np.nan
    #     try:
    #         a2 = getattr(a, 'foot2')
    #         ks2 = np.exp(-dist(a2, p)/(2*s*kapa**2))
    #     except HiddenKeypointError:
    #         flag2 = True
    #         ks2 = np.nan
    #     if flag1==True and flag2==True:
    #         return np.nan
    #     else:   
    #         return np.nanmax([ks1,ks2])


if __name__ == "__main__":
    main()