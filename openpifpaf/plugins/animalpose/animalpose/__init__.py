import openpifpaf

from . import animal_kp


def register():
    openpifpaf.DATAMODULES['animal'] = animal_kp.AnimalKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-animalpose'] = \
        "https://github.com/vita-epfl/openpifpaf-torchhub/releases/" \
        "download/v0.12.9/shufflenetv2k30-210511-120906-animal.pkl.epoch400"
