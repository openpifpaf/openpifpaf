"""Collections of decoders: fields to annotations."""

from . import utils
from .decoder import Decoder
from .cifcaf import CifCaf
from .cifdet import CifDet
from .pose_similarity import PoseSimilarity
from .tracking_pose import TrackingPose
from .factory import cli, configure, factory

from .factory import DECODERS
