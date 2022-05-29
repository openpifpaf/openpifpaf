import openpifpaf

from .cocokpst import CocoKpSt
from .posetrack2018 import Posetrack2018
from .posetrack2017 import Posetrack2017


def register():
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k16'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k30'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.12.10/'
        'tshufflenetv2k30-210628-075118-posetrack2018-cocokpst-slurm668247-o25-3d734bb8.pkl')

    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
    openpifpaf.DATAMODULES['posetrack2017'] = Posetrack2017
    openpifpaf.DATAMODULES['cocokpst'] = CocoKpSt
