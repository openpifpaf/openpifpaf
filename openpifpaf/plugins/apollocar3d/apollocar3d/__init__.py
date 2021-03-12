import openpifpaf

from . import apollo_kp


def register():
    openpifpaf.DATAMODULES['apollo'] = apollo_kp.ApolloKp
