import logging

from .encoder import Encoder
from .skeleton import Skeleton


def cli(parser):
    for encoder in Encoder.__subclasses__():
        encoder.cli(parser)


def factory(args, strides):
    for encoder in Encoder.__subclasses__():
        encoder.apply_args(args)

    headnames = args.headnets

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]
    if headnames[-1] == 'skeleton' and len(headnames) == len(strides) + 1:
        encoders.append(Skeleton())

    return encoders


def factory_head(head_name, stride):
    for encoder in Encoder.__subclasses__():
        logging.debug('checking whether encoder %s matches %s',
                      encoder.__name__, head_name)
        if not encoder.match(head_name):
            continue
        logging.info('selected encoder %s for %s', encoder.__name__, head_name)
        return encoder(head_name, stride)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
