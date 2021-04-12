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
from dataset_utilities.ds.instants_dataset import ViewCropperTransform
from dataset_utilities.ds.instants_dataset.instants_dataset import PlayerAnnotation

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

with OutputInhibitor():
    from openpifpaf.datasets.constants import COCO_KEYPOINTS
    from openpifpaf.datasets.deepsport import deepsportlab_dataset_splitter
    from openpifpaf.predict import main as predict


KiHEAD = 0.15
KiHIPS = 0.2
KiFEET = 0.1


class ScaleDownFactor2Transform():
    def __call__(self, view_key, view):
        view.image = view.image[::2,::2]
        return view

class HiddenKeypointError(BaseException):
    pass

class ScaleDownFactor2Transform():
    def __call__(self, view_key, view):
        view.image = view.image[::2,::2]
        return view


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_file")
    parser.add_argument("--pickled-dataset", required=True)
    parser.add_argument("--verbose", action='store_true', default=False)
    
    parser.add_argument('--adaptive-max-pool-th', action='store_true')
    parser.add_argument('--max-pool-th', default=0.1)
    parser.add_argument('--print-every', default=0, type=int)
    
    args = parser.parse_args()

    shape = (641,641)
    ds = PickledDataset("/data/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_ball_mask.pickle")
    ds = TransformedDataset(ds, [ViewCropperTransform(def_min=30, def_max=80, output_shape=shape, focus_object="player")])
    # ds = TransformedDataset(ds, [ScaleDownFactor2Transform()])
    keys = ds.keys.all()
    print("OKS metric computed on whole dataset since pose estimation is trained on COCO")
    result_list = []

    for k_index, key in enumerate(tqdm(keys)):
        view = ds.query_item(key)
        if not view:
            continue
        court = Court(view.rule_type)
        
        filename = f"{args.weights_file}_input_image.png"
        imageio.imwrite(filename, view.image)

        if args.adaptive_max_pool_th:
            sys.argv = [
            "aue",
            filename,
            "--checkpoint", args.weights_file, "--image-output",
            "--debug-images", "--debug-cif-c", "--debug", "--json-output",
            "--adaptive-max-pool-th",
            "--max-pool-th", args.max_pool_th,
            ]
        else:
            sys.argv = [
            "aue",
            filename,
            "--checkpoint", args.weights_file, "--image-output",
            "--debug-images", "--debug-cif-c", "--debug", "--json-output",
            "--max-pool-th", args.max_pool_th,
            ]
        

        with OutputInhibitor():
            predict()

        predictions = [PlayerSkeleton(**p) for p in json.load(open(f"{filename}.predictions.json", "r"))]
        predictions = [p for p in predictions if p.projects_in_court(view.calib, court) and p.visible]
        annotations = [PlayerAnnotation2D(a, view.calib) for a in view.annotations if a.type == "player" and a.camera == key.camera]

        matching = {}
        oks_list = []

        if predictions:
            for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
                if not annotations:
                    break
                idx = np.argmax([OKS(a, p) for a in annotations])
                matching[p] = annotations[idx]
                oks_list.append(OKS(annotations[idx], p, alpha=0.8))
                del annotations[idx]

        # remove remaining annotations that lie outside the court
        annotations = [a for a in annotations if a.projects_in_court(view.calib, court)]

        result_list.append({
            "predictions": predictions,
            "annotations": annotations + list(matching.values()),
            "matching": matching,
            "oks_list": oks_list,
        })

        if args.print_every > 0 and (k_index%args.print_every) == 0:
            pprint(compute_metrics(result_list))

    filename = f"{args.weights_file}_OKS_tmp_results.pickle"
    pickle.dump(result_list, open(filename, "wb"))
    pprint(compute_metrics(result_list))
    print(f"Temporary results have been saved in {filename}. To recompute the metric later, just call the 'compute_metrics' function on the content of that file.")

if __name__ == "__main__":
    main()