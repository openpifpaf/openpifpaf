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
                  head_meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            '{}.{}.c'.format(head_meta.dataset, head_meta.name),
            '{}.{}.vec'.format(head_meta.dataset, head_meta.name),
            '{}.{}.scales'.format(head_meta.dataset, head_meta.name),
        )

        self.bce_loss = components.BceL2()
        self.reg_loss = components.Regression()
        self.scale_loss = components.Scale()

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

        # determine foreground and background masks based on ground truth
        t = torch.transpose(t, 2, 4)
        finite = torch.isfinite(t)
        t_confidence_raw = t[:, :, :, :, 0:self.n_confidences]
        bg_mask = torch.all(t_confidence_raw == 0.0, dim=4)
        c_mask = torch.all(t_confidence_raw > 0.0, dim=4)
        reg_mask = torch.all(finite[:, :, :, :, self.n_confidences:1 + self.n_vectors * 2], dim=4)
        scale_mask = torch.all(finite[:, :, :, :, self.n_confidences + self.n_vectors * 3:], dim=4)

        # extract masked ground truth
        t_confidence_bg = t[bg_mask][:, 0:self.n_confidences]
        t_confidence = t[c_mask][:, 0:self.n_confidences]
        t_regs = t[reg_mask][:, self.n_confidences:1 + self.n_vectors * 2]
        t_sigma_min = t[reg_mask][
            :,
            self.n_confidences + self.n_vectors * 2:self.n_confidences + self.n_vectors * 3
        ]
        t_scales_reg = t[reg_mask][:, self.n_confidences + self.n_vectors * 3:]
        t_scales = t[scale_mask][:, self.n_confidences + self.n_vectors * 3:]

        # extract masked predictions
        x = torch.transpose(x, 2, 4)
        x_confidence_bg = x[bg_mask][:, 1:1 + self.n_confidences]
        x_logs2_c = x[c_mask][:, 0:1]
        x_confidence = x[c_mask][:, 1:1 + self.n_confidences]
        x_logs2_reg = x[reg_mask][:, 0:1]
        x_regs = x[reg_mask][:, 1 + self.n_confidences:1 + self.n_confidences + self.n_vectors * 2]
        # x_logs2_scale = x[scale_mask][:, 0:1]
        x_scales_reg = x[reg_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]
        x_scales = x[scale_mask][:, 1 + self.n_confidences + self.n_vectors * 2:]

        # impute t_scales_reg with predicted values
        t_scales_reg = t_scales_reg.clone()
        invalid_t_scales_reg = torch.isnan(t_scales_reg)
        t_scales_reg[invalid_t_scales_reg] = \
            torch.nn.functional.softplus(x_scales_reg.detach()[invalid_t_scales_reg])

        l_confidence_bg = self.bce_loss(x_confidence_bg, t_confidence_bg)
        l_confidence = self.bce_loss(x_confidence, t_confidence)
        l_reg = self.reg_loss(x_regs, t_regs, t_sigma_min, t_scales_reg)
        l_scale = self.scale_loss(x_scales, t_scales)

        # --- composite uncertainty
        # c
        x_logs2_c = 3.0 * torch.tanh(x_logs2_c / 3.0)
        l_confidence = 0.5 * l_confidence * torch.exp(-x_logs2_c) + 0.5 * x_logs2_c
        # reg
        x_logs2_reg = 3.0 * torch.tanh(x_logs2_reg / 3.0)
        # We want sigma = b*0.5. Therefore, log_b = 0.5 * log_s2 + log2
        x_logb = 0.5 * x_logs2_reg + 0.69314
        reg_factor = torch.exp(-x_logb)
        x_logb = x_logb.unsqueeze(1)
        reg_factor = reg_factor.unsqueeze(1)
        if self.n_vectors > 1:
            x_logb = torch.repeat_interleave(x_logb, self.n_vectors, 1)
            reg_factor = torch.repeat_interleave(reg_factor, self.n_vectors, 1)
        l_reg = l_reg * reg_factor + x_logb
        # scale
        # scale_factor = torch.exp(-x_logs2)
        # for i in range(self.n_scales):
        #     l_scale_component = l_scale[:, i]
        #     l_scale_component = l_scale_component * scale_factor + 0.5 * x_logs2

        if self.weights is not None:
            full_weights = torch.empty_like(t_confidence_raw)
            full_weights[:] = self.weights
            l_confidence_bg = full_weights[bg_mask] * l_confidence_bg
            l_confidence = full_weights[c_mask] * l_confidence
            l_reg = full_weights.unsqueeze(-1)[reg_mask] * l_reg
            l_scale = full_weights[scale_mask] * l_scale

        batch_size = t.shape[0]
        losses = [
            (torch.sum(l_confidence_bg) + torch.sum(l_confidence)) / batch_size,
            torch.sum(l_reg) / batch_size,
            torch.sum(l_scale) / batch_size,
        ]

        if not all(torch.isfinite(l).item() if l is not None else True for l in losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in losses]

        return losses
