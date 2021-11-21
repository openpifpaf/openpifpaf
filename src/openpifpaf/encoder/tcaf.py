import dataclasses
import logging
from typing import ClassVar, List, Tuple

from .. import headmeta
from .. import visualizer as visualizer_module
from .annrescaler import TrackingAnnRescaler
from .caf import CafGenerator

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Tcaf:
    """Tracking Composite Association Field."""

    meta: headmeta.Tcaf
    rescaler: TrackingAnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1
    visualizer: visualizer_module.Tcaf = None
    fill_plan: List[Tuple[int, int, int]] = None

    min_size: ClassVar[int] = 3
    fixed_size: ClassVar[bool] = True
    aspect_ratio: ClassVar[float] = 0.0
    padding: ClassVar[int] = 10

    def __post_init__(self):
        if self.rescaler is None:
            self.rescaler = TrackingAnnRescaler(self.meta.stride, self.meta.pose)

        if self.visualizer is None:
            self.visualizer = visualizer_module.Tcaf(self.meta)

        if self.fill_plan is None:
            self.fill_plan = [
                (caf_i, joint1i - 1, joint2i - 1)
                for caf_i, (joint1i, joint2i) in enumerate(self.meta.skeleton)
            ]

    def __call__(self, images, all_anns, metas):
        return CafGenerator(self)(images[0], all_anns, metas)
