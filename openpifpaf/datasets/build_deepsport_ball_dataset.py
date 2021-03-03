from tqdm.auto import tqdm

from dataset_utilities import AWSSession
from dataset_utilities.ds.instants_dataset import InstantsDataset, ViewsDataset, BuildCameraViews, AddBallAnnotation, DownloadFlags
from mlworkflow import FilteredDataset, TransformedDataset, PickledDataset

session = AWSSession(profile_name="gva@PROD").as_role("basketball-instants-dataset-ConsumerRole")

predicate = lambda instant_key, instant: len([a for a in instant.annotations if a.type == "player"]) > 0

instants_dataset = InstantsDataset(
    sport="basketball",
    local_storage="/DATA/datasets",
    session=session,
    predicate=predicate,
    progress_wrapper=tqdm,
    download_flags=DownloadFlags.WITH_CALIB_FILE|DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS|DownloadFlags.WITH_IMAGES
)

views_dataset = ViewsDataset(instants_dataset, view_builder=BuildCameraViews())
dataset = TransformedDataset(views_dataset, [AddBallAnnotation()])
dataset = FilteredDataset(dataset, predicate=lambda k,v: k.camera == v.ball.camera) # keep only cameras views in which there is a ball annotated. The rest is skipped.

PickledDataset.create(dataset, "/scratch/gva/camera_views_with_human_masks.pickle", yield_keys_wrapper=tqdm)
# python3 -m openpifpaf.train --lr=0.0001 --momentum=0.98  --epochs=150  --lr-decay 130 140  --lr-decay-epochs=10  --batch-size=32  --square-edge=385  --weight-decay=1e-5  --update-batchnorm-runningstatistics  --basenet=shufflenetv2k16w  --headnets cifball --dataset deepsport --deepsport-pickled-dataset /scratch/gva/views_camera_with_ball2.pickle
