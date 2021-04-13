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
KiHEAD = 0.15
KiHIPS = 0.2
KiFEET = 0.1

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

class PlayerAnnotation2D():
    def __init__(self, annotation: PlayerAnnotation, calib: Calib):
        self.annotation = annotation
        self.calib = calib
    @property
    def head(self):
        return self.calib.project_3D_to_2D(self.annotation.head)
    @property
    def hips(self):
        return self.calib.project_3D_to_2D(self.annotation.hips)
    @property
    def foot1(self):
        return self.calib.project_3D_to_2D(self.annotation.foot1)
    @property
    def foot2(self):
        return self.calib.project_3D_to_2D(self.annotation.foot2)

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
        FP = FP + np.sum(np.array(d["oks_list"])[np.newaxis] <  thresholds[:,np.newaxis], axis=1)
        Np = Np + len(predictions)
        Na = Na + len(annotations)
    return {
        "thresholds": thresholds,
        "precision": TP/Np,
        "recall": TP/Na,
        "Mprecision": np.mean(TP/Np),
        "Mrecall": np.mean(TP/Na)
    }

def dist(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))
    # return np.sum((p1-p2)**2)

def OKS(a: PlayerAnnotation2D, p: PlayerSkeleton, alpha=0.8):
    def KS(a, p, name, kapa, s, name2=None):
        name2 = name2 or name
        try: 
            return np.exp(-dist(getattr(a, name), getattr(p, name2))/(2*s*kapa**2))
        except HiddenKeypointError:
            return np.nan

    keypoints = Point2D([a.head, a.hips, a.foot1, a.foot2])

    # scale 
    s = (np.max(keypoints.x)-np.min(keypoints.x))*(np.max(keypoints.y)-np.min(keypoints.y)) # BB area in pixels

    pair1 = np.nanmean([KS(a, p, "head", KiHEAD*alpha, s), KS(a, p, "hips", KiHIPS*alpha, s), KS(a, p, "foot1", KiFEET*alpha, s), KS(a, p, "foot2", KiFEET*alpha, s)])
    pair2 = np.nanmean([KS(a, p, "head", KiHEAD*alpha, s), KS(a, p, "hips", KiHIPS*alpha, s), KS(a, p, "foot1", KiFEET*alpha, s, "foot2"), KS(a, p, "foot2", KiFEET*alpha, s, "foot1")])
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
    
    split = deepsportlab_dataset_splitter(keys, dataset_fold, 0, validation_set_size_pc)

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

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("weights_file")
    # args = parser.parse_args()
    args = cli()
    args.checkpoint = args.weights_file

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)
    
    assert args.dataset_fold == "DeepSport", "You should not eval OKS on another split than DeepSport"

    target_transforms = encoder.factory(model.head_nets, model.base_net.stride)
    # target_transforms = None
    heads = []
    for hd in model.head_nets:
        heads.append(hd.meta.name)

    data = build_DeepSport_test_dataset(
        pickled_dataset_filename=args.deepsport_pickled_dataset,
        validation_set_size_pc=15, square_edge=args.square_edge, target_transforms=target_transforms,
        preprocess=preprocess, focus_object=args.focus_object, config=heads, dataset_fold=args.dataset_fold)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True,
        # worker_init_fn=reset_pickled_datasets,
        collate_fn=collate_images_targets_inst_meta_views,)

    # shape = (641,641)
    # ds = PickledDataset("/data/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_ball_mask.pickle")
    # ds = TransformedDataset(ds, 
    #     [
    #         ViewCropperTransform(def_min=30, def_max=80, output_shape=shape, focus_object="player"),
    #         ExtractViewData(
    #         AddBallPositionFactory(),
    #         AddBallSegmentationTargetViewFactory(),
    #         AddHumansSegmentationTargetViewFactory(),)
    #         ])
    # keys = ds.keys.all()
    result_list = []

    for batch_i, (image_tensors_batch, target_batch, meta_batch, views_batch, keys_batch) in enumerate(tqdm(data_loader)):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device, target_batch=target_batch)
        
        # unbatch
        assert len(pred_batch)==len(views_batch)
        for pred, target, meta, view, key in zip(pred_batch, target_batch, meta_batch, views_batch, keys_batch):

            # print('pred len',len(pred))
            # print('pred',len(pred[0].shape))
            # print('pred',len(pred[1].shape))
            # print('pred',len(pred[2].shape))
            # raise

            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            # print(view.keys())
            court = Court(view['rule_type'])
            # print('Pred:', len([ann.json_data() for ann in pred]))
            # raise
            
            
            predictions = [PlayerSkeleton(**ann.json_data()) for ann in pred]
            predictions = [p for p in predictions if p.projects_in_court(view['calib'], court) and p.visible]
            annotations = [PlayerAnnotation2D(a, view['calib']) for a in view['annotations'] if a.type == "player" and a.camera == key.camera]
            # print('Predictions', len(predictions))
            # print('Annotations', len(annotations))
            if not predictions or not annotations:
                continue

            matching = {}
            oks_list = []
            for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
                if not annotations:
                    break
                idx = np.argmax([OKS(a, p) for a in annotations])
                matching[p] = annotations[idx]
                oks_list.append(OKS(annotations[idx], p, alpha=0.8))
                del annotations[idx]

            # remove remaining annotations that lie outside the court
            annotations = [a for a in annotations if a.projects_in_court(view['calib'], court)]

            result_list.append({
                "predictions": predictions,
                "annotations": annotations + list(matching.values()),
                "matching": matching,
                "oks_list": oks_list,
            })

        if batch_i%10 == 0:
            pprint(compute_metrics(result_list))


            #############################

            # # load the original image if necessary
            # cpu_image = None
            # if args.debug or args.show or args.image_output is not None:
            #     with open(meta['file_name'], 'rb') as f:
            #         cpu_image = PIL.Image.open(f).convert('RGB')

            # visualizer.BaseVisualizer.image(cpu_image)
            # if preprocess is not None:
            #     pred = preprocess.annotations_inverse(pred, meta)

            # if args.json_output is not None:
            #     json_out_name = out_name(
            #         args.json_output, meta['file_name'], '.predictions.json')
            #     LOG.debug('json output = %s', json_out_name)
            #     with open(json_out_name, 'w') as f:
            #         json.dump([ann.json_data() for ann in pred], f)

    # for k_index, key in enumerate(tqdm(keys)):
    #     view = ds.query_item(key)
    #     if not view:
    #         continue
    #     court = Court(view.rule_type)
        
    #     filename = f"{args.weights_file}_input_image.png"
    #     imageio.imwrite(filename, view.image)

    #     sys.argv = [
    #         "aue",
    #         filename,
    #         "--checkpoint", args.weights_file, "--image-output",
    #         "--debug-images", "--debug-cif-c", "--debug", "--json-output"
    #     ]

    #     with OutputInhibitor():
    #         predict()

    #     predictions = [PlayerSkeleton(**p) for p in json.load(open(f"{filename}.predictions.json", "r"))]
    #     predictions = [p for p in predictions if p.projects_in_court(view.calib, court) and p.visible]
    #     annotations = [PlayerAnnotation2D(a, view.calib) for a in view.annotations if a.type == "player" and a.camera == key.camera]
    #     if not predictions or not annotations:
    #         continue

    #     matching = {}
    #     oks_list = []
    #     for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
    #         if not annotations:
    #             break
    #         idx = np.argmax([OKS(a, p) for a in annotations])
    #         matching[p] = annotations[idx]
    #         oks_list.append(OKS(annotations[idx], p, alpha=0.8))
    #         del annotations[idx]

    #     # remove remaining annotations that lie outside the court
    #     annotations = [a for a in annotations if a.projects_in_court(view.calib, court)]

    #     result_list.append({
    #         "predictions": predictions,
    #         "annotations": annotations + list(matching.values()),
    #         "matching": matching,
    #         "oks_list": oks_list,
    #     })

    #     if k_index%10 == 0:
    #         pprint(compute_metrics(result_list))

    pickle.dump(result_list, open(f"{args.weights_file}_OKS_tmp_results.pickle", "wb"))
    pprint(compute_metrics(result_list))

if __name__ == "__main__":
    main()