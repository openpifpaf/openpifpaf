"""Convert a set of keypoint coordinates into target fields.

Takes an annotation from a dataset and turns it into the
ground truth for a field.
"""

from .annrescaler import AnnRescaler
from .annrescaler_ball import AnnRescalerBall
from .annrescaler_cent import AnnRescalerCent
from .factory import cli, configure, factory, factory_head
from .caf import Caf
from .cif import Cif
from .pan import PanopticTargetGenerator
