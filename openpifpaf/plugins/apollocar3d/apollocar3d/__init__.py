import openpifpaf

from . import apollo_kp


def register():
    openpifpaf.DATAMODULES['apollo'] = apollo_kp.ApolloKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-24'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/shufflenetv2k16-201113-135121-apollo.pkl.epoch290"
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-66'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/apollo_66kp-7d6ccbb9.pkl"
