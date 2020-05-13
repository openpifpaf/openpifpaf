import cProfile
import io
import logging
import pstats

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
            ps.dump_stats(self.out_name)
        print(iostream.getvalue())

        return result
