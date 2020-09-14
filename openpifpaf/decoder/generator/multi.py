import logging

from .generator import Generator

LOG = logging.getLogger(__name__)


class Multi(Generator):
    def __init__(self, generators):
        super().__init__()

        self.generators = generators

    def __call__(self, all_fields):
        out = []
        for task_i, generator in enumerate(self.generators):
            if generator is None:
                out.append(None)
                continue
            LOG.debug('task %d', task_i)
            out += generator(all_fields)

        return out

    def reset(self):
        for gen in self.generators:
            if not hasattr(gen, 'reset'):
                continue
            gen.reset()
