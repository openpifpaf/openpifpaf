import logging

import torch

LOG = logging.getLogger(__name__)


class ProfilerAutograd:
    trace_counter = 0

    def __init__(self, function_to_profile, *, device, out_name=None):
        if not out_name:
            out_name = 'pytorch_chrome_trace.json'

        self.function_to_profile = function_to_profile
        self.device = device

        self.out_name = out_name

    def __call__(self, *args, **kwargs):
        with torch.autograd.profiler.profile(use_cuda=str(self.device) == 'cuda') as prof:
            result = self.function_to_profile(*args, **kwargs)
        print(prof.key_averages())

        self.__class__.trace_counter += 1
        tracefilename = '{}.{}.json'.format(
            self.out_name.replace('.json', '').replace('.prof', ''),
            self.trace_counter,
        )
        LOG.info('writing trace file %s', tracefilename)
        prof.export_chrome_trace(tracefilename)

        return result
