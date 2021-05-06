import openpifpaf

from .cocokpst import CocoKpSt
from .posetrack2018 import Posetrack2018
from .posetrack2017 import Posetrack2017


def register():
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k16'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.2/tshufflenetv2k16-210228-220045-posetrack2018-cocokpst-o10-856584da.pkl'
    )
    # openpifpaf.CHECKPOINT_URLS['tshufflenetv2k30'] = (
    #     'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
    #     'v0.12.2/tshufflenetv2k30-210222-112623-posetrack2018-cocokpst-o10-123ec670.pkl'
    # )

    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
    openpifpaf.DATAMODULES['posetrack2017'] = Posetrack2017
    openpifpaf.DATAMODULES['cocokpst'] = CocoKpSt
