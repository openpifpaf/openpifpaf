"""Losses."""

import logging
import torch

from ..data import (
    COCO_PERSON_SIGMAS,
    COCO_PERSON_SKELETON,
    DENSER_COCO_PERSON_CONNECTIONS,
    DENSER_COCO_PERSON_SKELETON,
    KINEMATIC_TREE_SKELETON,
)

LOG = logging.getLogger(__name__)


def laplace_loss(x1, x2, logb, t1, t2, weight=None):
    """Loss based on Laplace Distribution.

    Loss for a single two-dimensional vector (x1, x2) with radial
    spread b and true (t1, t2) vector.
    """

    # left derivative of sqrt at zero is not defined, so prefer torch.norm():
    # https://github.com/pytorch/pytorch/issues/2421
    # norm = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    norm = (torch.stack((x1, x2)) - torch.stack((t1, t2))).norm(dim=0)

    losses = 0.694 + logb + norm * torch.exp(-logb)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def l1_loss(x1, x2, _, t1, t2, weight=None):
    """L1 loss.

    Loss for a single two-dimensional vector (x1, x2)
    true (t1, t2) vector.
    """
    losses = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
    if weight is not None:
        losses = losses * weight
    return torch.sum(losses)


def margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    max_r = torch.min((torch.stack(max_r1, max_r2, max_r3, max_r4)), axis=0)
    m0 = torch.isfinite(max_r)
    x = x[:, m0]
    t = t[:, m0]
    max_r = max_r[m0]

    # m1 = (x - t).norm(p=1, dim=0) > max_r
    # x = x[:, m1]
    # t = t[:, m1]
    # max_r = max_r[m1]

    norm = (x - t).norm(dim=0)
    m2 = norm > max_r

    return torch.sum(norm[m2] - max_r[m2])


def quadrant(xys):
    q = torch.zeros((xys.shape[1],), dtype=torch.long)
    q[xys[0, :] < 0.0] += 1
    q[xys[1, :] < 0.0] += 2
    return q


def quadrant_margin_loss(x1, x2, t1, t2, max_r1, max_r2, max_r3, max_r4):
    x = torch.stack((x1, x2))
    t = torch.stack((t1, t2))

    diffs = x - t
    qs = quadrant(diffs)
    norms = diffs.norm(dim=0)

    m1 = norms[qs == 0] > max_r1[qs == 0]
    m2 = norms[qs == 1] > max_r2[qs == 1]
    m3 = norms[qs == 2] > max_r3[qs == 2]
    m4 = norms[qs == 3] > max_r4[qs == 3]

    return (
        torch.sum(norms[qs == 0][m1] - max_r1[qs == 0][m1]) +
        torch.sum(norms[qs == 1][m2] - max_r2[qs == 1][m2]) +
        torch.sum(norms[qs == 2][m3] - max_r3[qs == 2][m3]) +
        torch.sum(norms[qs == 3][m4] - max_r4[qs == 3][m4])
    )


class SmoothL1Loss(object):
    def __init__(self, r_smooth, scale_required=True):
        self.r_smooth = r_smooth
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """
        if self.scale_required and self.scale is None:
            raise Exception
        if self.scale is None:
            self.scale = 1.0

        r = self.r_smooth * self.scale
        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < r

        smooth_loss = 0.5 / r[smooth_regime] * d[smooth_regime] ** 2
        linear_loss = d[smooth_regime == 0] - (0.5 * r[smooth_regime == 0])
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)


class MultiHeadLoss(torch.nn.Module):
    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses


class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    multiplicity_correction = False
    independence_scale = 3.0

    def __init__(self, head_name, regression_loss, *,
                 n_vectors, n_scales, sigmas=None, margin=False):
        super(CompositeLoss, self).__init__()

        self.n_vectors = n_vectors
        self.n_scales = n_scales
        if self.n_scales:
            assert len(sigmas) == n_scales
        if sigmas is None:
            sigmas = [[1.0] for _ in range(n_vectors)]
        if sigmas is not None:
            assert len(sigmas) == n_vectors
            scales_to_kp = torch.tensor(sigmas)
            scales_to_kp = torch.unsqueeze(scales_to_kp, 0)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            scales_to_kp = torch.unsqueeze(scales_to_kp, -1)
            self.register_buffer('scales_to_kp', scales_to_kp)
        else:
            self.scales_to_kp = None

        self.regression_loss = regression_loss or laplace_loss
        self.field_names = (
            ['{}.c'.format(head_name)] +
            ['{}.vec{}'.format(head_name, i + 1) for i in range(self.n_vectors)] +
            ['{}.scales{}'.format(head_name, i + 1) for i in range(self.n_scales)]
        )
        self.margin = margin
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

        LOG.debug('%s: n_vectors = %d, n_scales = %d, len(sigmas) = %d, margin = %s',
                  head_name, n_vectors, n_scales, len(sigmas), margin)

    def forward(self, *args):  # pylint: disable=too-many-statements
        x, t = args

        assert len(x) == 1 + 2 * self.n_vectors + self.n_scales
        x_intensity = x[0]
        x_regs = x[1:1 + self.n_vectors]
        x_spreads = x[1 + self.n_vectors:1 + 2 * self.n_vectors]
        x_scales = []
        if self.n_scales:
            x_scales = x[1 + 2 * self.n_vectors:1 + 2 * self.n_vectors + self.n_scales]

        assert len(t) == 1 + self.n_vectors + 1
        target_intensity = t[0]
        target_regs = t[1:1 + self.n_vectors]
        target_scale = t[-1]

        bce_masks = (target_intensity[:, :-1] + target_intensity[:, -1:]) > 0.5
        if not torch.any(bce_masks):
            return None, None, None

        batch_size = x_intensity.shape[0]
        LOG.debug('batch size = %d', batch_size)

        bce_x_intensity = x_intensity
        bce_target_intensity = target_intensity[:, :-1]
        if self.bce_blackout:
            bce_x_intensity = bce_x_intensity[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            bce_target_intensity = bce_target_intensity[:, self.bce_blackout]

        LOG.debug('BCE: target = %s, mask = %s', bce_target_intensity.shape, bce_masks.shape)
        bce_target = torch.masked_select(bce_target_intensity, bce_masks)
        bce_weight = None
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target)
            bce_weight[bce_target == 0] = self.background_weight
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.masked_select(bce_x_intensity, bce_masks),
            bce_target,
            weight=bce_weight,
        )

        reg_losses = [None for _ in target_regs]
        reg_masks = target_intensity[:, :-1] > 0.5
        if torch.any(reg_masks):
            weight = None
            if self.multiplicity_correction:
                assert len(target_regs) == 2
                lengths = torch.norm(target_regs[0] - target_regs[1], dim=2)
                multiplicity = (lengths - 3.0) / self.independence_scale
                multiplicity = torch.clamp(multiplicity, min=1.0)
                multiplicity = torch.masked_select(multiplicity, reg_masks)
                weight = 1.0 / multiplicity

            reg_losses = []
            for i, (x_reg, x_spread, target_reg) in enumerate(zip(x_regs, x_spreads, target_regs)):
                if hasattr(self.regression_loss, 'scale'):
                    assert self.scales_to_kp is not None
                    self.regression_loss.scale = torch.masked_select(
                        torch.clamp(target_scale * self.scales_to_kp[i], 0.1, 1000.0),  # pylint: disable=unsubscriptable-object
                        reg_masks,
                    )

                reg_losses.append(self.regression_loss(
                    torch.masked_select(x_reg[:, :, 0], reg_masks),
                    torch.masked_select(x_reg[:, :, 1], reg_masks),
                    torch.masked_select(x_spread, reg_masks),
                    torch.masked_select(target_reg[:, :, 0], reg_masks),
                    torch.masked_select(target_reg[:, :, 1], reg_masks),
                    weight=weight,
                ) / 1000.0 / batch_size)

        scale_losses = []
        if x_scales:
            scale_losses = [
                torch.nn.functional.l1_loss(
                    torch.masked_select(x_scale, reg_masks),
                    torch.masked_select(target_scale * scale_to_kp, reg_masks),
                    reduction='sum',
                ) / 1000.0 / batch_size
                for x_scale, scale_to_kp in zip(x_scales, self.scales_to_kp)
            ]

        margin_losses = [None for _ in target_regs] if self.margin else []
        if self.margin and torch.any(reg_masks):
            margin_losses = []
            for i, (x_reg, target_reg) in enumerate(zip(x_regs, target_regs)):
                margin_losses.append(quadrant_margin_loss(
                    torch.masked_select(x_reg[:, :, 0], reg_masks),
                    torch.masked_select(x_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 0], reg_masks),
                    torch.masked_select(target_reg[:, :, 1], reg_masks),
                    torch.masked_select(target_reg[:, :, 2], reg_masks),
                    torch.masked_select(target_reg[:, :, 3], reg_masks),
                    torch.masked_select(target_reg[:, :, 4], reg_masks),
                    torch.masked_select(target_reg[:, :, 5], reg_masks),
                ) / 100.0 / batch_size)

        return [ce_loss] + reg_losses + scale_losses + margin_losses


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=[30.0, 2.0, 2.0, 50.0, 3.0, 3.0],
                       type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--r-smooth', type=float, default=0.0,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
    group.add_argument('--background-weight', default=1.0, type=float,
                       help='[experimental] BCE weight of background')
    group.add_argument('--paf-multiplicity-correction',
                       default=False, action='store_true',
                       help='[experimental] use multiplicity correction for PAF loss')
    group.add_argument('--paf-independence-scale', default=3.0, type=float,
                       help='[experimental] linear length scale of independence for PAF regression')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')


def factory_from_args(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight
    CompositeLoss.fixed_size = args.paf_fixed_size
    CompositeLoss.aspect_ratio = args.paf_aspect_ratio

    return factory(
        args.headnets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        r_smooth=args.r_smooth,
        device=args.device,
        margin=args.margin_loss,
    )


def loss_parameters(head_name):
    n_vectors = None
    if 'pif' in head_name:
        n_vectors = 1
    elif 'paf' in head_name:
        n_vectors = 2

    n_scales = None
    if 'pif' in head_name:
        n_scales = 1
    elif 'paf' in head_name:
        n_scales = 0

    sigmas = None
    if head_name == 'pif':
        sigmas = [COCO_PERSON_SIGMAS]
    elif head_name in ('paf', 'paf19', 'wpaf'):
        sigmas = [
            [COCO_PERSON_SIGMAS[j1i - 1] for j1i, _ in COCO_PERSON_SKELETON],
            [COCO_PERSON_SIGMAS[j2i - 1] for _, j2i in COCO_PERSON_SKELETON],
        ]
    elif head_name in ('paf16',):
        sigmas = [
            [COCO_PERSON_SIGMAS[j1i - 1] for j1i, _ in KINEMATIC_TREE_SKELETON],
            [COCO_PERSON_SIGMAS[j2i - 1] for _, j2i in KINEMATIC_TREE_SKELETON],
        ]
    elif head_name in ('paf44',):
        sigmas = [
            [COCO_PERSON_SIGMAS[j1i - 1] for j1i, _ in DENSER_COCO_PERSON_SKELETON],
            [COCO_PERSON_SIGMAS[j2i - 1] for _, j2i in DENSER_COCO_PERSON_SKELETON],
        ]
    elif head_name in ('paf25',):
        sigmas = [
            [COCO_PERSON_SIGMAS[j1i - 1] for j1i, _ in DENSER_COCO_PERSON_CONNECTIONS],
            [COCO_PERSON_SIGMAS[j2i - 1] for _, j2i in DENSER_COCO_PERSON_CONNECTIONS],
        ]

    return {
        'n_vectors': n_vectors,
        'n_scales': n_scales,
        'sigmas': sigmas,
    }


def factory(head_names, lambdas, *,
            reg_loss_name=None, r_smooth=None, device=None, margin=False):
    if isinstance(head_names[0], (list, tuple)):
        return [factory(hn, lam,
                        reg_loss_name=reg_loss_name,
                        r_smooth=r_smooth,
                        device=device,
                        margin=margin)
                for hn, lam in zip(head_names, lambdas)]

    head_names = [h for h in head_names if h not in ('skeleton', 'tskeleton')]

    if reg_loss_name == 'smoothl1':
        reg_loss = SmoothL1Loss(r_smooth)
    elif reg_loss_name == 'l1':
        reg_loss = l1_loss
    elif reg_loss_name == 'laplace':
        reg_loss = laplace_loss
    elif reg_loss_name is None:
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    losses = [CompositeLoss(head_name, reg_loss,
                            margin=margin, **loss_parameters(head_name))
              for head_name in head_names]
    loss = MultiHeadLoss(losses, lambdas)

    if device is not None:
        loss = loss.to(device)

    return loss
