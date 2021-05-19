"""An open implementation of PifPaf."""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .annotation import Annotation, AnnotationDet
from .configurable import Configurable
from .predictor import Predictor
from . import datasets
from . import decoder
from . import encoder
from . import headmeta
from . import logger
from . import metric
from . import network
from . import optimize
from . import plugin
from . import visualizer

from .datasets import DATAMODULES
from .decoder import DECODERS
from .network.factory import (
    BASE_FACTORIES,
    BASE_TYPES,
    CHECKPOINT_URLS,
    HEADS,
    PRETRAINED_UNAVAILABLE,
)
from .network.losses.factory import LOSSES, LOSS_COMPONENTS
from .network.nets import MODEL_MIGRATION
from .show.annotation_painter import PAINTERS

# load plugins last
plugin.register()
