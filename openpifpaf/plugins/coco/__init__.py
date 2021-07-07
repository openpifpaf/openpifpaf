import openpifpaf

from .cocodet import CocoDet
from .cocokp import CocoKp
from .dataset import CocoDataset


def register():
    openpifpaf.DATAMODULES['cocodet'] = CocoDet
    openpifpaf.DATAMODULES['cocokp'] = CocoKp

    # human pose estimation
    openpifpaf.CHECKPOINT_URLS['mobilenetv2'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12a5/mobilenetv2-201112-193315-cocokp-1728a9f5.pkl')
    openpifpaf.CHECKPOINT_URLS['mobilenetv3large'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.9/mobilenetv3large-210426-191110-cocokp-o10s-8d9bfba6.pkl')
    openpifpaf.CHECKPOINT_URLS['resnet18'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['resnet50'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.2/resnet50-210224-202010-cocokp-o10s-d020d7f1.pkl')
    openpifpaf.CHECKPOINT_URLS['resnet101'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['resnet152'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2x1'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2x2'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.6/shufflenetv2k16-210404-110105-cocokp-o10s-f90ed364.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-withdense'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12b4/shufflenetv2k16-210221-131426-cocokp-o10s-627d901e.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.6/shufflenetv2k30-210409-024202-cocokp-o10s-f4fb0807.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k44'] = openpifpaf.PRETRAINED_UNAVAILABLE

    # object detection
    openpifpaf.CHECKPOINT_URLS['resnet18-cocodet'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/'
        'v0.12.10/resnet18-210526-031303-cocodet-slurm610002-1faf5801.pkl')
