import logging
import re

from .encoder import Encoder
from .paf import Paf
from .pif import Pif
from .skeleton import Skeleton

LOG = logging.getLogger(__name__)


def cli(parser):
    for encoder in Encoder.__subclasses__():
        encoder.cli(parser)


def factory(args, strides):
    for encoder in Encoder.__subclasses__():
        encoder.apply_args(args)

    return factory_heads(args.headnets, strides)


def factory_heads(headnames, strides):
    if isinstance(headnames[0], (list, tuple)):
        return [factory_heads(task_headnames, task_strides)
                for task_headnames, task_strides in zip(headnames, strides)]

    encoders = [factory_head(head_name, stride)
                for head_name, stride in zip(headnames, strides)]
    if headnames[-1] == 'skeleton' and len(headnames) == len(strides) + 1:
        encoders.append(Skeleton())

    return encoders


def factory_head(head_name, stride):
    if head_name in ('pif',
                     'ppif',
                     'pifb',
                     'pifs') or \
       re.match('pif([0-9]+)$', head_name) is not None:
        LOG.info('selected encoder Pif for %s', head_name)
        return Pif(head_name, stride)

    if head_name in ('paf',
                     'pafs',
                     'wpaf',
                     'pafb') or \
       re.match('paf([0-9]+)$', head_name) is not None:
        LOG.info('selected encoder Paf for %s', head_name)
        return Paf(head_name, stride)

    raise Exception('unknown head to create an encoder: {}'.format(head_name))
