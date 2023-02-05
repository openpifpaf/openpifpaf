import argparse
import logging
from typing import Dict, List

import torch

from . import components

LOG = logging.getLogger(__name__)


class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""

    @classmethod
    def factory_from_headmeta(cls, head_meta):
        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_meta.name, head_meta.n_vectors, head_meta.n_scales)

        weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)
            LOG.debug("The weights for the keypoints are %s", weights)

        loss_components: Dict[str, List[components.Base]] = {
            f'{head_meta.dataset}.{head_meta.name}.c': [components.Bce([1], [0], weights=weights)],
        }

        regression_components: List[components.Base] = []
        if head_meta.n_vectors <= head_meta.n_scales:
            # keypoints and associations: vectors matched with scales and can
            # have additional scales
            regression_components = [
                components.Regression(
                    [
                        2 + vi * 2,
                        2 + vi * 2 + 1,
                        2 + head_meta.n_vectors * 2 + vi,
                    ],
                    [
                        1 + vi * 2,
                        1 + vi * 2 + 1,
                        1 + head_meta.n_vectors * 2 + vi,
                        1 + head_meta.n_vectors * 3 + vi,
                    ],
                    weights=weights,
                )
                for vi in range(head_meta.n_vectors)
            ]
        elif head_meta.n_vectors == 2 and head_meta.n_scales == 0:
            # detection
            regression_components = [
                components.Regression(
                    [
                        2 + vi * 2,
                        2 + vi * 2 + 1,
                        2 + 1 * 2,  # width
                        2 + 1 * 2 + 1,  # height
                    ],
                    [
                        1 + vi * 2,
                        1 + vi * 2 + 1,
                        1 + 2 * 2 + vi,
                        1 + 1 * 2,  # width
                        1 + 1 * 2 + 1,  # height
                    ],
                    weights=weights,
                    sigma_from_scale=0.1,  # for detection
                    scale_from_wh=True,  # for detection
                )
                for vi in range(head_meta.n_vectors)
            ]

        if regression_components:
            loss_components[f'{head_meta.dataset}.{head_meta.name}.vec'] = regression_components

        if head_meta.n_scales:
            loss_components[f'{head_meta.dataset}.{head_meta.name}.scales'] = [
                components.Scale(
                    [2 + head_meta.n_vectors * 2 + si],
                    [1 + head_meta.n_vectors * 3 + si],
                    weights=weights,
                )
                for si in range(head_meta.n_scales)
            ]

        return cls(loss_components)

    def __init__(
        self,
        loss_components: Dict[str, List[components.Base]],
    ):
        super().__init__()
        self.loss_components = loss_components

        self.previous_losses = None

    @property
    def field_names(self):
        return self.loss_components.keys()

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    def configure(cls, args: argparse.Namespace):
        pass

    # pylint: disable=too-many-statements
    def forward(self, x, t):
        LOG.debug('loss for %s', self.field_names)

        if t is None:
            return [None for _ in self.loss_components]

        # run all loss components
        x = torch.transpose(x, 2, 4)
        t = torch.transpose(t, 2, 4)
        losses = {
            name: [l(x, t) for l in loss_components]
            for name, loss_components in self.loss_components.items()
        }

        batch_size = t.shape[0]
        losses = [
            None
            if not ls
            else (torch.sum(ls[0] if len(ls) == 1 else torch.cat(ls))) / batch_size
            for ls in losses.values()
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
