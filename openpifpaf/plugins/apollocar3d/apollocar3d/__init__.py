import openpifpaf

from . import apollo_kp


def register():
    openpifpaf.DATAMODULES['apollo'] = apollo_kp.ApolloKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-24'] = None   # TODO
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-66'] = None   # TODO
