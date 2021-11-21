from .basenetworks import BaseNetwork
from .running_cache import RunningCache
from ..signal import Signal


class TrackingBase(BaseNetwork):
    cached_items = [0, -1]

    def __init__(self, single_image_backbone):
        super().__init__(
            't' + single_image_backbone.name,
            stride=single_image_backbone.stride,
            out_features=single_image_backbone.out_features,
        )
        self.single_image_backbone = single_image_backbone
        self.running_cache = RunningCache(self.cached_items)

        Signal.subscribe('eval_reset', self.reset)

    def reset(self):
        del self.running_cache
        self.running_cache = RunningCache(self.cached_items)

    def forward(self, *args):
        x = args[0]

        # backbone
        x = self.single_image_backbone(x)

        # feature cache
        if not self.training:
            x = self.running_cache(x)

        return x
