import openpifpaf

from .cocodet import CocoDet
from .cocokp import CocoKp
from .dataset import CocoDataset


def register():
    openpifpaf.DATAMODULES['cocodet'] = CocoDet
    openpifpaf.DATAMODULES['cocokp'] = CocoKp
