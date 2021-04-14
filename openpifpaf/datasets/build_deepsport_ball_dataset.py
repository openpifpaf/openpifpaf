from tqdm.auto import tqdm

from dataset_utilities.providers import AWSSession
from dataset_utilities.ds.instants_dataset import InstantsDataset, ViewsDataset, BuildCameraViews, AddBallAnnotation, DownloadFlags
from mlworkflow import FilteredDataset, TransformedDataset, PickledDataset

session = AWSSession(profile_name="abolfazl@ucl").as_role("basketball-instants-dataset-ConsumerRole")

# # predicate = lambda instant_key, instant: len([a for a in instant.annotations if a.type == "player"]) > 0
# predicate = lambda instant_key, instant: instant.annotated_human_masks

# instants_dataset = InstantsDataset(
#     sport="basketball",
#     local_storage="/scratch/mistasse/abolfazl/keemotion",
#     session=session,
#     predicate=predicate,
#     progress_wrapper=tqdm,
#     download_flags=DownloadFlags.WITH_CALIB_FILE|DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS|DownloadFlags.WITH_IMAGES
# )

# # instants_dataset.download()
# views_dataset = ViewsDataset(instants_dataset, view_builder=BuildCameraViews())
# dataset = TransformedDataset(views_dataset, [AddBallAnnotation()])
# # dataset = FilteredDataset(dataset, predicate=lambda k,v: k.camera == v.ball.camera) # keep only cameras views in which there is a ball annotated. The rest is skipped.

# # dataset = FilteredDataset(dataset, predicate=lambda k,v: k.camera == v.ball.camera and v.human_masks is not None) # keep only cameras views for which there's a human masks and the ball
# # dataset = FilteredDataset(dataset, predicate=lambda k,v: v.human_masks is not None) # keep only cameras views for which there's a human masks


# # with open("/data/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_large.pickle", "wb") as f:
# with open("/scratch/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_ball_mask.pickle", "wb") as f:
#     PickledDataset.create(dataset, f, yield_keys_wrapper=tqdm)
# # python3 -m openpifpaf.train --lr=0.0001 --momentum=0.98  --epochs=150  --lr-decay 130 140  --lr-decay-epochs=10  --batch-size=32  --square-edge=385  --weight-decay=1e-5  --update-batchnorm-runningstatistics  --basenet=shufflenetv2k16w  --headnets cifball --dataset deepsport --deepsport-pickled-dataset /scratch/gva/views_camera_with_ball2.pickle



# predicate = lambda instant_key, instant: instant.annotated_human_masks
# df = DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS | DownloadFlags.WITH_IMAGES | DownloadFlags.WITH_CALIB_FILE
# instants_dataset = InstantsDataset(sport="basketball", session=session,
#     local_storage="/scratch/mistasse/abolfazl/keemotion", progress_wrapper=tqdm,
#     predicate=predicate, download_flags = df)
# # build a dataset of balls centered in the image with a margin of 50cm around the ball
# views_dataset = ViewsDataset(instants_dataset, view_builder=BuildCameraViews())
# ds = TransformedDataset(views_dataset, [AddBallAnnotation()])
# PickledDataset.create(ds, "/scratch/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_ball_mask.pickle")




# predicate = lambda instant_key, instant: instant.annotated_human_masks
df = DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS | DownloadFlags.WITH_IMAGES | DownloadFlags.WITH_CALIB_FILE
instants_dataset = InstantsDataset(sport="basketball", session=session,
    local_storage="/scratch/mistasse/abolfazl/keemotion", progress_wrapper=tqdm,
    download_flags = df,
    predicate=lambda k,v: v.annotated_human_masks)
# build a dataset of balls centered in the image with a margin of 50cm around the ball
views_dataset = ViewsDataset(instants_dataset, view_builder=BuildCameraViews())
ds = TransformedDataset(views_dataset, [AddBallAnnotation()])
# ds = FilteredDataset(ds, predicate=lambda k,v: k.camera == v.ball.camera)
PickledDataset.create(ds, "/scratch/mistasse/abolfazl/keemotion/pickled/camera_views_with_human_masks_ball_mask_v2.pickle")
