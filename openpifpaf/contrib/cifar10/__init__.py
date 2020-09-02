import openpifpaf

from . import datamodule


def register():
    openpifpaf.DATAMODULES['cifar10'] = datamodule.Cifar10
