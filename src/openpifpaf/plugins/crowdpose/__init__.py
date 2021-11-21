import openpifpaf

from .module import CrowdPose


def register():
    openpifpaf.CHECKPOINT_URLS['resnet50-crowdpose'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12a7/resnet50-201005-100758-crowdpose-d978a89f.pkl'
    )

    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose
