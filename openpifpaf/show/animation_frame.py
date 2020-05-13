import logging

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.pyplot as plt
    import matplotlib.patches
except ImportError:
    matplotlib = None
    plt = None


LOG = logging.getLogger(__name__)


class AnimationFrame:
    video_fps = 10
    video_dpi = 100

    def __init__(self, *,
                 fig_width=8.0,
                 fig_init_args=None,
                 show=False,
                 video_output=None,
                 second_visual=False):
        self.fig_width = fig_width
        self.fig_init_args = fig_init_args or {}
        self.show = show
        self.video_output = video_output
        self.video_writer = None
        if self.video_output:
            self.video_writer = matplotlib.animation.writers['ffmpeg'](fps=self.video_fps)

        self.second_visual = second_visual
        if self.second_visual:
            self.fig_width *= 2

        if plt is None:
            LOG.error('matplotlib is not installed')

        self.fig = None
        self.ax = None
        self.ax_second = None
        self._skip_frame = False

        LOG.info('video output = %s', video_output)

    @staticmethod
    def clean_axis(ax):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def skip_frame(self):
        self._skip_frame = True

    def iter(self):
        video_writer_is_setup = False
        try:
            while True:
                yield (self.ax, self.ax_second)

                if self._skip_frame:
                    self._skip_frame = False
                    continue

                # Lazy setup of video writer (needs to be after first yield
                # because only that might setup `self.fig`).
                if self.video_writer and not video_writer_is_setup:
                    self.video_writer.setup(self.fig, self.video_output, self.video_dpi)
                    video_writer_is_setup = True

                if self.show:
                    plt.pause(0.01)
                if self.video_writer:
                    self.video_writer.grab_frame()

                if self.ax:
                    self.clean_axis(self.ax)
                if self.ax_second:
                    self.clean_axis(self.ax_second)

        finally:
            if self.video_writer:
                self.video_writer.finish()
            if self.fig:
                plt.close(self.fig)

    def frame_init(self, image):
        if plt is None:
            return None, None

        if 'figsize' not in self.fig_init_args:
            self.fig_init_args['figsize'] = (
                self.fig_width,
                self.fig_width * image.shape[0] / image.shape[1]
            )
            if self.second_visual:
                self.fig_init_args['figsize'] = (
                    self.fig_init_args['figsize'][0],
                    self.fig_init_args['figsize'][1] / 2.0,
                )

        self.fig = plt.figure(**self.fig_init_args)
        if self.second_visual:
            self.ax = plt.Axes(self.fig, [0.0, 0.0, 0.5, 1.0])
            self.ax_second = plt.Axes(self.fig, [0.5, 0.05, 0.45, 0.9])
            self.fig.add_axes(self.ax)
            self.fig.add_axes(self.ax_second)
        else:
            self.ax = plt.Axes(self.fig, [0.0, 0.0, 1.0, 1.0])
            self.ax_second = None
            self.fig.add_axes(self.ax)
        self.ax.set_axis_off()
        # self.ax.set_xlim(0, image.shape[1])
        # self.ax.set_ylim(image.shape[0], 0)
        if self.ax_second is not None:
            self.ax_second.set_axis_off()
            # self.ax_second.set_xlim(0, image.shape[1])
            # self.ax_second.set_ylim(image.shape[0], 0)

        return self.ax, self.ax_second
