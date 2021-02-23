import openpifpaf

from . import datamodule
from . import basenet


def register():
    openpifpaf.DATAMODULES['cifar10'] = datamodule.Cifar10

    openpifpaf.BASE_TYPES.add(basenet.Cifar10Net)
    openpifpaf.BASE_FACTORIES['cifar10net'] = lambda: basenet.Cifar10Net()  # pylint: disable=unnecessary-lambda
