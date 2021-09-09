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
    openpifpaf.CHECKPOINT_URLS['mobilenetv3small'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'mobilenetv3small-210822-213409-cocokp-slurm726252-edge513-o10s-803b24ae.pkl')
    openpifpaf.CHECKPOINT_URLS['mobilenetv3large'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'mobilenetv3large-210820-184901-cocokp-slurm725985-edge513-o10s-6c76cbfb.pkl')
    openpifpaf.CHECKPOINT_URLS['resnet18'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['resnet50'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'resnet50-210830-150728-cocokp-slurm728641-edge513-o10s-ecd30da4.pkl')
    openpifpaf.CHECKPOINT_URLS['resnet101'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['resnet152'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2x1'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2x2'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'shufflenetv2k16-210820-232500-cocokp-slurm726069-edge513-o10s-7189450a.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-withdense'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12b4/shufflenetv2k16-210221-131426-cocokp-o10s-627d901e.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'shufflenetv2k30-210821-003923-cocokp-slurm726072-edge513-o10s-5fe1c400.pkl')
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k44'] = openpifpaf.PRETRAINED_UNAVAILABLE
    openpifpaf.CHECKPOINT_URLS['swin_s'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_s_fpn_lvl_3_lr_5e-5_resumed-d286d41a.pkl')
    openpifpaf.CHECKPOINT_URLS['swin_b'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_b_fpn_lvl_3_lr_5e-5_resumed-fa951ce0.pkl')
    openpifpaf.CHECKPOINT_URLS['swin_t_input_upsample'] = (
        'http://github.com/dmizr/openpifpaf/releases/download/'
        'v0.12.14/swin_t_input_upsample_no_fpn_lr_5e-5_resumed-e0681112.pkl')

    # object detection
    openpifpaf.CHECKPOINT_URLS['mobilenetv3small-cocodet'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/v0.13/'
        'mobilenetv3small-210822-215020-cocodet-slurm726253-5f2c894f.pkl')
    openpifpaf.CHECKPOINT_URLS['resnet18-cocodet'] = (
        'http://github.com/openpifpaf/torchhub/releases/download/'
        'v0.12.10/resnet18-210526-031303-cocodet-slurm610002-1faf5801.pkl')
