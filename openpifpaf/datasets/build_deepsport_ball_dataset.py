from dataset_utilities import AWSSession
from dataset_utilities.ds.instants_dataset import InstantsDataset, ViewsDataset, BuildCameraViews, AddBallAnnotation
from mlworkflow import FilteredDataset, TransformedDataset, PickledDataset
from tqdm.notebook import tqdm

session = AWSSession("image-processing-developer")

instants_dataset = InstantsDataset(sport="basketball", local_storage="/DATA/datasets", session=session)
views_dataset = ViewsDataset(instants_dataset, view_builder=BuildCameraViews())
dataset = TransformedDataset(views_dataset, [AddBallAnnotation()])
dataset = FilteredDataset(dataset, predicate=lambda k,v: k.camera == v.ball.camera)

PickledDataset.create(dataset, "/scratch/gva/views_camera_with_ball2.pickle", yield_keys_wrapper=tqdm)
# python3 -m openpifpaf.train --lr=0.0001 --momentum=0.98  --epochs=150  --lr-decay 130 140  --lr-decay-epochs=10  --batch-size=32  --square-edge=385  --weight-decay=1e-5  --update-batchnorm-runningstatistics  --basenet=shufflenetv2k16w  --headnets cifball --dataset deepsport --deepsport-pickled-dataset /scratch/gva/views_camera_with_ball2.pickle
