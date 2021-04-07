import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # needs to be placed before any 'import tensorflow' statement
import sys
import json
import pandas
import subprocess
import imageio
import numpy as np
import pickle
from pprint import pprint
from tqdm.auto import tqdm

from mlworkflow import PickledDataset, TransformedDataset


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
device_index = 0
visible_devices = [physical_devices[device_index]] # one single visible device for now
tf.config.set_visible_devices(visible_devices, device_type="GPU")
for device in visible_devices:
    tf.config.experimental.set_memory_growth(device, enable=True)



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



class ComputeKeypointsDetectionMetrics(ChunkProcessor):
    def __init__(self, non_max_suppression_pool_size=50, threshold=0.5, target_enlargment_size=10):
        if isinstance(threshold, np.ndarray):
            thresholds = threshold
        elif isinstance(threshold, list):
            thresholds = np.array(threshold)
        elif isinstance(threshold, float):
            thresholds = np.array([threshold])
        else:
            raise ValueError(f"Unsupported type for input argument 'threshold'. Recieved {threshold}")
        assert len(thresholds.shape) == 1, "'threshold' argument should be 1D-array (a scalar is also accepted)."

        self.avoid_local_eq = AvoidLocalEqualities()
        self.peak_local_max = PeakLocalMax(min_distance=non_max_suppression_pool_size//2, thresholds=thresholds)
        self.enlarge_target = tf.keras.layers.MaxPool2D(target_enlargment_size, strides=1, padding="same")
        self.compute_metric = ComputeElementaryMetrics()

    def __call__(self, chunk):
        batch_target = tf.cast(chunk["batch_target"], tf.float32)
        batch_output = chunk["batch_heatmap"]
        assert len(batch_target.shape) == len(batch_output.shape)
        batch_target = batch_target if len(batch_target.shape) == 4 else batch_target[...,tf.newaxis]
        batch_output = batch_output if len(batch_output.shape) == 4 else batch_output[...,tf.newaxis]

        batch_hitmap = self.peak_local_max(self.avoid_local_eq(batch_output))
        batch_hitmap = tf.cast(batch_hitmap, tf.int32)
        batch_target = self.enlarge_target(batch_target)
        batch_target = tf.cast(batch_target, tf.int32)[..., tf.newaxis]

        batch_metric = self.compute_metric(batch_hitmap=batch_hitmap, batch_target=batch_target)
        chunk["batch_hitmap"] = tf.cast(batch_hitmap, tf.float32)*batch_output[..., tf.newaxis]
        chunk["batch_TP"] = batch_metric["batch_TP"]
        chunk["batch_FP"] = batch_metric["batch_FP"]
        chunk["batch_TN"] = batch_metric["batch_TN"]
        chunk["batch_FN"] = batch_metric["batch_FN"]

class ComputeKeypointsTopKDetectionMetrics():
    def __init__(self, k):
        self.k = k if isinstance(k, list) else [k]
    def __call__(self, chunk):
        assert len(chunk["batch_target"].get_shape()) == 3, \
            "Only one keypoint type is allowed. If 'batch_target' is one_hot encoded, it needs to be compressed before."
        _, H, W, C, N = shape = [tf.shape(chunk["batch_hitmap"])[k] for k in range(5)]

        batch_target = tf.cast(chunk["batch_target"], tf.int32)
        shape = [-1, C, N, H*W]
        flatten_hitmap = tf.reshape(tf.transpose(chunk["batch_hitmap"], perm=[0,3,4,1,2]), shape=shape)
        values, indices = tf.math.top_k(flatten_hitmap, k=max(self.k), sorted=True)
        gather_indices = np.array(self.k)-1 # k=[1,2,10] corresponds to indices [0,1,9]
        chunk["topk_outputs"] = tf.gather(values, gather_indices, axis=-1)
        chunk["topk_indices"] = tf.gather(tf.stack(((indices // W), (indices % W)), -1), gather_indices, axis=-2)
        chunk["topk_targets"] = tf.gather_nd(batch_target, chunk["topk_indices"], batch_dims=1)

        chunk["batch_P"] = tf.cast(tf.reduce_any(batch_target!=0, axis=[1,2]), tf.int32)
        chunk["batch_N"] = 1-chunk["batch_P"]

        #"batch_TP" not in chunk or logging.warning("'batch_TP' is being overwritten") # pylint: disable=expression-not-assigned
        #"batch_FP" not in chunk or logging.warning("'batch_FP' is being overwritten") # pylint: disable=expression-not-assigned

        chunk["batch_topk_TP"] = tf.cast(tf.cast(tf.math.cumsum(chunk["topk_targets"], axis=-1), tf.bool), tf.int32)
        chunk["batch_topk_FP"] = tf.cast(tf.cast(chunk["topk_outputs"], tf.bool), tf.int32)-chunk["topk_targets"]

def divide(num: np.ndarray, den: np.ndarray):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den>0)


class Callback():
    precedence = 10
    events = ["epoch_begin", "cycle_begin", "batch_begin", "batch_end", "cycle_end", "epoch_end"]
    def fire(self, event, state):
        assert event in self.events, f"Unknown event: {event}. Available events are {self.events}"
        cb = getattr(self, "on_{}".format(event), None)
        if cb:
            cb(**state, state=state) # pylint: disable=not-callable

class InitState(Callback):
    precedence = 0 # very first
    def on_epoch_begin(self, state, **_):
        for key in [k for k in state if k!= "epoch"]:
            state[key] = np.nan

class AccumulateBatchMetrics(Callback):
    def __init__(self, **metrics):
        """ Accumulate metrics output per batch by the network.
            Arguments:
                metrics - A dictionary of metrics to accumulate as pairs of
                    (input_name, output_name) where `input_name` is is the
                    [B, ...] metric name as output by the network, and
                    `output_name` is the [...] metric in which elements were
                    summed over the batch dimension
        """
        self.metrics = metrics
    def on_cycle_begin(self, **_):
        self.acc = {}
    def on_batch_end(self, state, **_):
        for input_name, output_name in self.metrics.items():
            value = np.sum(state[input_name], axis=0)
            self.acc[output_name] = self.acc.setdefault(output_name, np.zeros_like(value)) + value
    def on_cycle_end(self, state, **_): # 'state' attribute in R/W
        for input_name, output_name in self.metrics.items():
            # Remove temporary metric from state dictionary
            state.pop(input_name)
            # Record metric to state dictionary
            state[output_name] = self.acc[output_name]

class ComputeMetrics(Callback):
    precedence = 50
    def __init__(self, thresholds=np.linspace(0,1,51), class_index=0):
        self.class_index = class_index
        self.thresholds = thresholds
    def on_cycle_end(self, state, **_):
        TP = state["TP"][self.class_index]
        FP = state["FP"][self.class_index]
        TN = state["TN"][self.class_index]
        FN = state["FN"][self.class_index]

        data = {
            "thresholds": self.thresholds,
            "accuracy": (TP+TN)/(TP+TN+FP+FN),
            "precision": divide(TP, TP+FP),
            "recall": divide(TP, TP+FN),
            "TP rate": divide(TP, TP+FN),
            "FP rate": divide(FP, FP+TN),
        }
        state["metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

        state.pop("TP")
        state.pop("FP")
        state.pop("TN")
        state.pop("FN")

class ComputeTopkMetrics(Callback):
    precedence = 50
    def __init__(self, k, thresholds, class_index):
        self.k = k if isinstance(k, list) else [k]
        self.thresholds = thresholds
        self.class_index = class_index
    def on_cycle_end(self, state, **_):
        for i, k in enumerate(self.k):
            FP = state["topk_FP"][self.class_index, :, i]
            TP = state["topk_TP"][self.class_index, :, i]
            P = state["P"][np.newaxis]
            N = state["N"][np.newaxis]
            data = {
                "thresholds": self.thresholds,
                f"top{k} FP rate": divide(FP, P + N),  # #-possible cases is the number of images
                f"top{k} TP rate": divide(TP, P),      # #-possible cases is the number of images on which there's a ball to detect
                f"top{k} precision": divide(TP, TP + FP),
                f"top{k} recall": divide(TP, P),
            }
            state[f"top{k}_metrics"] = pandas.DataFrame(np.vstack([data[name] for name in data]).T, columns=list(data.keys()))

        # TODO: handle multple classes cases (here only class index is picked and the rest is discarded)
        state.pop("topk_FP")
        state.pop("topk_TP")
        state.pop("N")
        state.pop("P")


def infer(weights_file, data, basenet, headnets, verbose=False):
    image_filename = "image/tmp_image_ball_metric.png"
    imageio.imwrite(image_filename, data["input_image"])
    basenet_kwargs = [] if basenet is None else ["--basenet", basenet]
    headnets_kwargs = [] if headnets is None else ["--headnets", *headnets]
    sys.argv = [
        "aue",
        image_filename,
        "--checkpoint", weights_file,
        *basenet_kwargs,
        *headnets_kwargs,
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
    parser.add_argument("--pickled-dataset-of-masks", required=True)
    parser.add_argument("--pickled-dataset-of-balls", required=True)
    parser.add_argument("--basenet")
    parser.add_argument("--headnets", nargs="*")
    parser.add_argument("--dataset-split", required=True)
    parser.add_argument("--test-fold", default=0)
    parser.add_argument("--verbose", action='store_true', default=False)
    args = parser.parse_args()

    # DATASET
    shape = (641,641)
    ds = PickledDataset(args.pickled_dataset_of_balls)
    ds = TransformedDataset(ds, [
        ViewCropperTransform(def_min=30, def_max=80, output_shape=shape),
        ExtractViewData(AddBallPositionFactory(), AddBallSegmentationTargetViewFactory()),
    ])

    human_masks_keys = list(PickledDataset(args.pickled_dataset_of_masks).keys.all())
    split = deepsportlab_dataset_splitter(human_masks_keys, args.dataset_split, args.test_fold, validation_set_size_pc=15)
    split["testing"] = [k for k in ds.keys.all() if k not in human_masks_keys]
    split["training"] = [k for k in ds.keys.all() if k in split["training"]]
    split["validation"] = [k for k in ds.keys.all() if k in split["validation"]]

    # TENSORFLOW METRIC EVALUATION MODEL
    chunk = {}
    chunk["batch_heatmap"] = tf.keras.Input(dtype=tf.uint8, shape=(shape[1], shape[0]), name="batch_heatmap")
    chunk["batch_target"] = tf.keras.Input(dtype=tf.uint8, shape=(shape[1], shape[0]), name="batch_target")
    inputs = dict(chunk) # makes a copy

    n_points = 21
    thresholds = np.linspace(0,1,n_points)
    k = [1]
    chunk_processors = [
        CastFloat(["batch_heatmap", "batch_target"]),
        Normalize("batch_heatmap"),
        ComputeKeypointsDetectionMetrics(threshold=thresholds),
        ComputeKeypointsTopKDetectionMetrics(k=k),
    ]
    for cp in chunk_processors:
        cp(chunk)


    callbacks = [
        InitState(),
        AccumulateBatchMetrics(batch_TP="TP", batch_FP="FP", batch_TN="TN", batch_FN="FN", batch_topk_FP="topk_FP", batch_topk_TP="topk_TP", batch_P="P", batch_N="N"),
        ComputeMetrics(class_index=0, thresholds=thresholds),
        ComputeTopkMetrics(class_index=0, thresholds=thresholds, k=k),
    ]

    outputs = {k:chunk[k] for k in chunk if k in ["batch_TP", "batch_FP", "batch_TN", "batch_FN",
        "batch_topk_TP", "batch_topk_FP", "batch_P", "batch_N"]}
    model = tf.keras.Model(inputs, outputs)

    state = {}
    subsets_results_dict = {}
    for subset_name, keys in split.items():
        state["subset"] = subset_name
        _ = [cb.fire("cycle_begin", state) for cb in callbacks]
        for k_index, key in enumerate(tqdm(keys)):
            data = ds.query_item(key)
            if data is None:
                continue

            heatmap = infer(args.weights_file, data, args.basenet, args.headnets, args.verbose)

            _ = [cb.fire("batch_begin", state) for cb in callbacks]
            result = model({"batch_heatmap": heatmap[np.newaxis], "batch_target": data["mask"][np.newaxis]})
            state.update(**result)
            _ = [cb.fire("batch_end", state) for cb in callbacks]
        _ = [cb.fire("cycle_end", state) for cb in callbacks]
        subsets_results_dict[subset_name] = state.copy()

    filename = f"{args.weights_file}_ball_tmp_results.pickle"
    pickle.dump(subsets_results_dict, open(filename, "wb"))
    print(f"Temporary results have been saved in {filename}. To recompute the metric later, just call the 'plot_roc' or 'plot_f1' functions on the content of that file.")

if __name__ == "__main__":
    main()