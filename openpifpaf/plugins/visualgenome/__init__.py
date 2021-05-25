from .datamodule import VGModule
import openpifpaf

def register():
    openpifpaf.DATAMODULES['vg'] = VGModule
