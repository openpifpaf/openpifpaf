import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # needs to be placed before any 'import tensorflow' statement
import sys
import json
import subprocess
import imageio
import numpy as np
import pickle
from pprint import pprint
from tqdm.auto import tqdm

from mlworkflow import PickledDataset, TransformedDataset

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
    from openpifpaf.datasets.deepsport import AddBallSegmentationTargetViewFactory, AddBallPositionFactory, deepsportlab_dataset_splitter
    import tensorflow as tf
    from tf_layers import AvoidLocalEqualities, PeakLocalMax, ComputeElementaryMetrics
    from openpifpaf.predict import main as predict
    from openpifpaf.datasets.constants import COCO_KEYPOINTS


from dataset_utilities.court import Court
from dataset_utilities.calib import Calib, Point2D
from dataset_utilities.ds.instants_dataset import ExtractViewData, ViewCropperTransform




class ChunkProcessor:
    pass
class CastFloat(ChunkProcessor):
    def __init__(self, tensor_name):
        self.tensor_name = [tensor_name] if isinstance(tensor_name, str) else tensor_name
    def __call__(self, chunk):
        for tensor_name in self.tensor_name:
            if tensor_name in chunk:
                chunk[tensor_name] = tf.cast(chunk[tensor_name], tf.float32)
class Normalize(ChunkProcessor):
    def __init__(self, tensor_name):
        self.tensor_name = tensor_name
    def __call__(self, chunk):
        assert chunk[self.tensor_name].dtype == tf.float32
        chunk[self.tensor_name] = chunk[self.tensor_name]/255
class ComputeKeypointsDetectionAccuracy(ChunkProcessor):
    def __init__(self, non_max_suppression_pool_size=50, threshold=0.5, target_enlargment_size=10):
        thresholds = threshold if isinstance(threshold, np.ndarray) else np.array([threshold])
        assert len(thresholds.shape) == 1, "'threshold' argument should be 1D-array (a scalar is also accepted)."

        self.avoid_local_eq = AvoidLocalEqualities()
        self.peak_local_max = PeakLocalMax(min_distance=non_max_suppression_pool_size//2, thresholds=thresholds)
        self.enlarge_target = tf.keras.layers.MaxPool2D(target_enlargment_size, strides=1, padding="same")
        self.compute_metric = ComputeElementaryMetrics()

    def __call__(self, chunk):
        batch_target = tf.cast(chunk["batch_target"], tf.float32)
        batch_target = batch_target if len(batch_target.shape) == 4 else batch_target[...,tf.newaxis]
        batch_output = chunk["batch_heatmap"]
        batch_output = batch_output if len(batch_output.shape) == 4 else batch_output[...,tf.newaxis]

        batch_output = self.avoid_local_eq(batch_output)
        batch_hitmap = self.peak_local_max(batch_output)
        batch_hitmap = tf.cast(batch_hitmap, tf.int32)
        chunk["batch_hitmap"] = tf.squeeze(batch_hitmap)
        batch_target = self.enlarge_target(batch_target)
        batch_target = tf.cast(batch_target, tf.int32)[..., tf.newaxis]

        batch_metric = self.compute_metric(batch_hitmap=batch_hitmap, batch_target=batch_target)
        chunk["batch_TP"] = tf.squeeze(batch_metric["batch_TP"])
        chunk["batch_FP"] = tf.squeeze(batch_metric["batch_FP"])
        chunk["batch_TN"] = tf.squeeze(batch_metric["batch_TN"])
        chunk["batch_FN"] = tf.squeeze(batch_metric["batch_FN"])

        



def infer(weights_file, data, verbose=False):
    image_filename = "image/tmp_image_ball_metric.png"
    imageio.imwrite(image_filename, data["input_image"])
    sys.argv = [
        "aue",
        image_filename,
        "--checkpoint", weights_file,
        "--image-output",
        "--debug-images", "--debug-cif-c", "--debug"
    ]
    if verbose:
        predict()
    else:
        with OutputInhibitor():
            predict()
    return imageio.imread("image/test.accumulated.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_file")
    parser.add_argument("--pickled-dataset", required=True)
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--test-fold", default=0)
    parser.add_argument("--verbose", action='store_true', default=False)
    args = parser.parse_args()


    # DATASET
    shape = (641,641)
    ds = PickledDataset(args.pickled_dataset)
    ds = TransformedDataset(ds, [
        ViewCropperTransform(def_min=30, def_max=80, output_shape=shape),
        ExtractViewData(AddBallPositionFactory(), AddBallSegmentationTargetViewFactory()),
    ])
    split = deepsportlab_dataset_splitter(list(ds.keys.all()), args.dataset_split, args.test_fold)


    # TENSORFLOW METRIC EVALUATION MODEL
    chunk = {}
    chunk["batch_heatmap"] = tf.keras.Input(dtype=tf.uint8, shape=(shape[1], shape[0]), name="batch_heatmap")
    chunk["batch_target"] = tf.keras.Input(dtype=tf.uint8, shape=(shape[1], shape[0]), name="batch_target")
    inputs = dict(chunk) # makes a copy

    thresholds = np.array([])
    n_points = 21
    chunk_processors = [
        CastFloat(["batch_heatmap", "batch_target"]),
        Normalize("batch_heatmap"),
        ComputeKeypointsDetectionAccuracy(non_max_suppression_pool_size=20, threshold=np.linspace(0,1,n_points)),
    ]
    for cp in chunk_processors:
        cp(chunk)

    outputs = {k:chunk[k] for k in chunk if k in ["batch_TP", "batch_TN", "batch_FP", "batch_FN"]}
    model = tf.keras.Model(inputs, outputs)


    subsets_results_dict = {}
    for subset_name, keys in split.items():
        current_result = subsets_results_dict[subset_name] = {}
        current_result["TP"] = np.zeros(n_points, np.int32)
        current_result["FP"] = np.zeros(n_points, np.int32)
        current_result["TN"] = np.zeros(n_points, np.int32)
        current_result["FN"] = np.zeros(n_points, np.int32)

        for k_index, key in enumerate(tqdm(keys)):
            data = ds.query_item(key)
            if data is None:
                continue

            heatmap = infer(args.weights_file, data, args.verbose)

            result = model({"batch_heatmap": heatmap[np.newaxis], "batch_target": data["mask"][np.newaxis]})
            current_result["TP"] += result["batch_TP"].numpy()
            current_result["FP"] += result["batch_FP"].numpy()
            current_result["TN"] += result["batch_TN"].numpy()
            current_result["FN"] += result["batch_FN"].numpy()

    filename = f"{args.weights_file}_ball_tmp_results.pickle"
    pickle.dump(subsets_results_dict, open(filename, "wb"))
    print(f"Temporary results have been saved in {filename}. To recompute the metric later, just call the 'plot_roc' or 'plot_f1' functions on the content of that file.")

if __name__ == "__main__":
    main()