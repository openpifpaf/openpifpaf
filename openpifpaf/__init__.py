"""An open implementation of PifPaf."""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .annotation import Annotation, AnnotationDet
from . import datasets
from . import decoder
from . import network
from . import optimize
