from .datamodule import VGModule
import openpifpaf


def register():
    openpifpaf.DATAMODULES['visualgenome'] = VGModule
