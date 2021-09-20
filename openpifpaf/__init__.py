"""An open implementation of PifPaf."""

# pylint: disable=wrong-import-position

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# register ops first
from . import cpp_extension
cpp_extension.register_ops()

from .annotation import Annotation, AnnotationDet
from .configurable import Configurable
from .predictor import Predictor
from .signal import Signal
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
from .network.model_migration import MODEL_MIGRATION
from .show.annotation_painter import PAINTERS

# load plugins last
plugin.register()
