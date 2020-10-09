"""An open implementation of PifPaf."""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .annotation import Annotation, AnnotationDet
from . import datasets
from . import decoder
from . import headmeta
from . import logger
from . import metric
from . import network
from . import optimize
from . import plugins

from .datasets import DATAMODULES
from .decoder import DECODERS
from .network.factory import BASE_FACTORIES, BASE_TYPES, HEAD_FACTORIES, HEAD_TYPES
from .network.nets import MODEL_MIGRATION
