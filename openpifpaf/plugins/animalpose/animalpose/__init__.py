import openpifpaf

from . import animal_kp


def register():
    openpifpaf.DATAMODULES['animal'] = animal_kp.AnimalKp
