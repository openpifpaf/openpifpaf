import cProfile
import io
import logging
import pstats

import torch

LOG = logging.getLogger(__name__)


class Profiler:
    def __init__(self, function_to_profile, *, profile=None, out_name=None):
        if profile is None:
            profile = cProfile.Profile()

        self.function_to_profile = function_to_profile
        self.profile = profile
        self.out_name = out_name

    def __call__(self, *args, **kwargs):
        self.profile.enable()

        result = self.function_to_profile(*args, **kwargs)

        self.profile.disable()
        iostream = io.StringIO()
        ps = pstats.Stats(self.profile, stream=iostream)
        ps = ps.sort_stats('tottime')
        ps.print_stats()
        if self.out_name:
            LOG.info('writing profile file %s', self.out_name)
            ps.dump_stats(self.out_name)
        print(iostream.getvalue())

        return result


class TorchProfiler:
    trace_counter = 0
    table_sort_by = 'self_cpu_time_total'

    def __init__(self, function_to_profile, *, out_name=None):
        if not out_name:
            out_name = 'torchprofiler_chrome_trace.json'

        self.function_to_profile = function_to_profile
        self.out_name = out_name

    def __call__(self, *args, **kwargs):
        with torch.profiler.profile(with_stack=True) as prof:
            result = self.function_to_profile(*args, **kwargs)
        print(prof.key_averages().table(sort_by=self.table_sort_by, row_limit=-1))

        self.__class__.trace_counter += 1
        tracefilename = '{}.{}.json'.format(
            self.out_name.replace('.json', '').replace('.prof', ''),
            self.trace_counter,
        )
        LOG.info('writing trace file %s', tracefilename)
        prof.export_chrome_trace(tracefilename)

        return result
