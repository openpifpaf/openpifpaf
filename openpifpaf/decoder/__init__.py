"""Collections of decoders: fields to annotations."""

from . import utils
from .decoder import Decoder
from .cifcaf import CifCaf
from .cifdet import CifDet
from .factory import cli, configure, factory
from .profiler import Profiler, TorchProfiler

from .factory import DECODERS
