import logging
from typing import List

from .decoder import Decoder

LOG = logging.getLogger(__name__)


class Multi(Decoder):
    def __init__(self, decoders):
        super().__init__()

        self.decoders = decoders

    def __call__(self, all_fields):
        out = []
        for task_i, decoder in enumerate(self.decoders):
            if decoder is None:
                out.append(None)
                continue
            LOG.debug('task %d', task_i)
            out += decoder(all_fields)

        return out

    def reset(self):
        # TODO: remove?
        for dec in self.decoders:
            if not hasattr(dec, 'reset'):
                continue
            dec.reset()

    @classmethod
    def factory(cls, head_metas) -> List['Generator']:
        raise NotImplementedError
