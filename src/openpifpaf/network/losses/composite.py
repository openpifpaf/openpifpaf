import argparse
import logging

import torch

from . import components

LOG = logging.getLogger(__name__)


class CompositeLoss(torch.nn.Module):
    """Default loss since v0.13"""

    def __init__(self, head_meta):
        super().__init__()
        self.n_confidences = head_meta.n_confidences
        self.n_vectors = head_meta.n_vectors
        self.n_scales = head_meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_meta.name, head_meta.n_vectors, head_meta.n_scales)

        if head_meta.n_vectors == head_meta.n_scales:
            # keypoints and associations: vectors matched with scales
            regression_components = [
                components.Regression(
                    [
                        2 + vi * 2,
                        2 + vi * 2 + 1,
                        2 + self.n_vectors * 2 + vi,
                    ],
                    [
                        1 + vi * 3,
                        1 + vi * 3 + 1,
                        1 + vi * 3 + 2,
                        1 + self.n_vectors * 3 + vi,
                    ],
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
                        1 + vi * 3,
                        1 + vi * 3 + 1,
                        1 + vi * 3 + 2,
                        1 + 1 * 3,  # width
                        1 + 1 * 3 + 1,  # height
                    ],
                    sigma_from_scale=0.1,  # for detection
                    scale_from_wh=True,  # for detection
                )
                for vi in range(head_meta.n_vectors)
            ]
        else:
            regression_components = []

        self.loss_components = {
            f'{head_meta.dataset}.{head_meta.name}.c': [components.Bce([1], [0])],
            f'{head_meta.dataset}.{head_meta.name}.vec': regression_components,
            f'{head_meta.dataset}.{head_meta.name}.scales': [
                components.Scale(
                    [2 + head_meta.n_vectors * 2 + si],
                    [1 + head_meta.n_vectors * 3 + si],
                )
                for si in range(head_meta.n_scales)
            ]
        }

        self.weights = None
        if head_meta.training_weights is not None:
            assert len(head_meta.training_weights) == head_meta.n_fields
            self.weights = torch.Tensor(head_meta.training_weights).reshape(1, -1, 1, 1, 1)

        LOG.debug("The weights for the keypoints are %s", self.weights)
        self.bce_blackout = None
        self.previous_losses = None

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
            return [None, None, None]
        assert x.shape[2] == 1 + self.n_confidences + self.n_vectors * 2 + self.n_scales
        assert t.shape[2] == self.n_confidences + self.n_vectors * 3 + self.n_scales

        # run all loss components
        x = torch.transpose(x, 2, 4)
        t = torch.transpose(t, 2, 4)
        losses = {
            name: [l(x, t) for l in loss_components]
            for name, loss_components in self.loss_components.items()
        }

        if self.weights is not None:
            full_weights = torch.empty_like(t[:, :, :, :, 0])
            full_weights[:] = self.weights
            l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
            l_confidence = full_weights[c_mask] * l_confidence
            l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
            l_scale = full_weights[scale_mask] * l_scale

        batch_size = t.shape[0]
        losses = [
            (ls[0] if len(ls) == 1 else torch.sum(torch.cat(ls))) / batch_size
            for ls in losses.values()
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
