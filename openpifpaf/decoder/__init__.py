"""Collections of decoders: fields to annotations."""

from .caf_scored import CafScored
from .cif_hr import CifHr, CifDetHr
from .cif_seeds import CifSeeds
from .factory import cli, configure, factory_decode, factory_from_args
from .field_config import FieldConfig
from .generator.cifcaf import CifCaf
from .generator.cifdet import CifDet
from .generator.generator import Generator
from . import nms
from .occupancy import Occupancy
from .profiler import Profiler
from .profiler_autograd import ProfilerAutograd
