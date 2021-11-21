"""Convert a set of keypoint coordinates into target fields.

Takes an annotation from a dataset and turns it into the
ground truth for a field.
"""

from .annrescaler import AnnRescaler, AnnRescalerDet, TrackingAnnRescaler
from .factory import cli, configure
from .caf import Caf
from .cif import Cif
from .cifdet import CifDet
from .single_image import SingleImage
from .tcaf import Tcaf
