
import openpifpaf

from . import animal_kp

from .animal_kp import AnimalKp


def register():
    openpifpaf.DATAMODULES['animal'] = animal_kp.AnimalKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-animalpose'] = \
        "http://github.com/vita-epfl/openpifpaf-torchhub/releases/" \
        "download/v0.12.9/shufflenetv2k30-210511-120906-animal.pkl.epoch400"
