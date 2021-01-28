"""Losses."""

import logging
import torch

from . import heads

from torch.autograd import Variable
import torch.nn.functional as F

import os.path
from os import path

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

    # constrain range of logb
    logb = 3.0 * torch.tanh(logb / 3.0)

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


def logl1_loss(logx, t, **kwargs):
    """Swap in replacement for functional.l1_loss."""
    return torch.nn.functional.l1_loss(
        logx, torch.log(t), **kwargs)


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
    r_smooth = 0.0

    def __init__(self, *, scale_required=True):
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

## AMA

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels, delta=1.):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    errors = (delta - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super(MultiHeadLoss, self).__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        # print('1111111111111111111')
        # print('multihead loss: %s, %s', self.field_names, self.lambdas)
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0  # TODO implement
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        # print('222222222222222222222222')
        # print(len(self.lambdas))
        # print('333333333333333333333333')
        # print(len(flat_head_losses))
        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses


class MultiHeadLossAutoTune(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head less.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        In the common setting, use lambdas of zero and one to deactivate and
        activate the tasks you want to train. Less common, if you have
        secondary tasks, you can reduce their importance by choosing a
        lambda value between zero and one.
        """
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas
        self.sparse_task_parameters = sparse_task_parameters

        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float64),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.log_sigmas.exp()]}

    def forward(self, *args):
        head_fields, head_targets = args
        LOG.debug('losses = %d, fields = %d, targets = %d',
                  len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.log_sigmas) == len(flat_head_losses)
        loss_values = [lam * l / (2.0 * (log_sigma.exp() ** 2))
                       for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                       if l is not None]
        auto_reg = [lam * log_sigma
                    for lam, log_sigma, l in zip(self.lambdas, self.log_sigmas, flat_head_losses)
                    if l is not None]
        total_loss = sum(loss_values) + sum(auto_reg) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses


class CompositeLoss(torch.nn.Module):
    background_weight = 1.0
    focal_gamma = 1.0
    margin = False

    def __init__(self, head_net: heads.CompositeField, regression_loss):
        super(CompositeLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d, margin = %s',
                  head_net.meta.name, self.n_vectors, self.n_scales, self.margin)

        self.regression_loss = regression_loss or laplace_loss
        self.field_names = (
            ['{}.c'.format(head_net.meta.name)] +
            ['{}.vec{}'.format(head_net.meta.name, i + 1) for i in range(self.n_vectors)] +
            ['{}.scales{}'.format(head_net.meta.name, i + 1) for i in range(self.n_scales)]
        )
        if self.margin:
            self.field_names += ['{}.margin{}'.format(head_net.meta.name, i + 1)
                                 for i in range(self.n_vectors)]

        self.bce_blackout = None

    def _confidence_loss(self, x_confidence, target_confidence):
        bce_masks = torch.isnan(target_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        # TODO assumes one confidence
        # print('x_confidence')
        # print(x_confidence.shape)
        x_confidence = x_confidence[:, :, 0]

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            target_confidence = target_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, target_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(target_confidence, bce_masks)
        bce_weight = 1.0
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        if self.background_weight != 1.0:
            bce_weight = torch.ones_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 0] *= self.background_weight
        elif self.focal_gamma != 0.0:
            bce_weight = torch.empty_like(bce_target, requires_grad=False)
            bce_weight[bce_target == 1] = x_confidence[bce_target == 1]
            bce_weight[bce_target == 0] = -x_confidence[bce_target == 0]
            bce_weight = (1.0 + torch.exp(bce_weight)).pow(-self.focal_gamma)
        ce_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            x_confidence,
            bce_target,
            # weight=bce_weight,
            reduction='none',
        ) * bce_weight).sum() / (1000.0 * batch_size)

        return ce_loss

    def _localization_loss(self, x_regs, x_logbs, target_regs):
        batch_size = target_regs[0].shape[0]
        # print('localization loss')
        # print(x_regs.shape)
        # print(x_logbs.shape)
        # print(len(target_regs))
        # print(target_regs[0].shape)

        reg_losses = []
        for i, target_reg in enumerate(target_regs):
            reg_masks = torch.isnan(target_reg[:, :, 0]).bitwise_not_()
            if not torch.any(reg_masks):
                reg_losses.append(None)
                continue

            reg_losses.append(self.regression_loss(
                torch.masked_select(x_regs[:, :, i, 0], reg_masks),
                torch.masked_select(x_regs[:, :, i, 1], reg_masks),
                torch.masked_select(x_logbs[:, :, i], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                weight=0.1,
            ) / (100.0 * batch_size))

        return reg_losses

    @staticmethod
    def _scale_losses(x_scales, target_scales):
        assert x_scales.shape[2] == len(target_scales)
        # print('scale_losess')
        # print(x_scales.shape)
        # print(len(target_scales))
        # print(target_scales[0].shape)

        batch_size = x_scales.shape[0]
        return [
            logl1_loss(
                torch.masked_select(x_scales[:, :, i], torch.isnan(target_scale).bitwise_not_()),
                torch.masked_select(target_scale, torch.isnan(target_scale).bitwise_not_()),
                reduction='sum',
            ) / (100.0 * batch_size)
            for i, target_scale in enumerate(target_scales)
        ]

    def _margin_losses(self, x_regs, target_regs, *, target_confidence):
        if not self.margin:
            return []

        # print('margin losses')
        # print(x_regs.shape)
        # print(len(target_regs))
        # print(target_regs[0].shape)
        # print(len(target_confidence))
        # print(target_confidence[0].shape)

        reg_masks = target_confidence > 0.5
        if not torch.any(reg_masks):
            return [None for _ in target_regs]

        batch_size = reg_masks.shape[0]
        margin_losses = []
        for x_reg, target_reg in zip(x_regs, target_regs):
            margin_losses.append(quadrant_margin_loss(
                torch.masked_select(x_reg[:, :, 0], reg_masks),
                torch.masked_select(x_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 0], reg_masks),
                torch.masked_select(target_reg[:, :, 1], reg_masks),
                torch.masked_select(target_reg[:, :, 2], reg_masks),
                torch.masked_select(target_reg[:, :, 3], reg_masks),
                torch.masked_select(target_reg[:, :, 4], reg_masks),
                torch.masked_select(target_reg[:, :, 5], reg_masks),
            ) / (100.0 * batch_size))
        return margin_losses

    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)

        x, t = args
        # print('hell yes')
        # print(t)
        # print(len(t))
        # t, mask = t

        x = [xx.double() for xx in x]
        t = [tt.double() for tt in t]
        # print('pif instances')
        # print(len(t))
        # print(t[0].shape)
        # print(mask.shape)
        # print(t.shape)

        x_confidence, x_regs, x_logbs, x_scales = x

        assert len(t) == 1 + self.n_vectors + self.n_scales
        running_t = iter(t)
        target_confidence = next(running_t)
        target_regs = [next(running_t) for _ in range(self.n_vectors)]
        target_scales = [next(running_t) for _ in range(self.n_scales)]

        ce_loss = self._confidence_loss(x_confidence, target_confidence)
        reg_losses = self._localization_loss(x_regs, x_logbs, target_regs)
        scale_losses = self._scale_losses(x_scales, target_scales)
        margin_losses = self._margin_losses(x_regs, target_regs,
                                            target_confidence=target_confidence)
        # print('++++++++++++++++++++++++PIF OUTPUT')
        # print([ce_loss] + reg_losses + scale_losses + margin_losses)
        # print(ce_loss)
        # print(reg_losses)
        # print(scale_losses)
        # print(margin_losses)
        return [ce_loss] + reg_losses + scale_losses + margin_losses

### AMA
class SegmantationLoss(torch.nn.Module):
    def __init__(self, head_net: heads.InstanceSegHead, device):
        super(SegmantationLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_sigmas = head_net.meta.n_sigmas

        self.field_names = [1,1,1]
        self.device = device

        

    def _meshgrid(self):
        y = torch.arange(0,1024.)/1024
        x = torch.arange(0,2048.)/1024
        meshgrid = torch.stack([
            y[:,None].expand(-1,2048),
            x[None,:].expand(1024,-1)
        ], dim=0).cuda()

        return meshgrid

    @staticmethod
    def _masked(tensor, mask):
        return tensor[mask.expand_as(tensor)].view(tensor.shape[0], -1)
    
    @staticmethod
    def _unravel_index(index, shape):
        res = []
        for size in shape[::-1]:
            res.append(index % size)
            index = index // size
        return tuple(res[::-1])

    def unique_ids(self, b):
        """unique ids in an image, background included if present"""
        return self.instances(b).unique().tolist()

    def instance_ids(self, b):
        """instances ids in an image, background excluded"""
        ids = self.unique_ids(b)
        if ids[0] == 0:
            ids = ids[1:]

        # print('func instance_ids')
        # print(b)
        # print(ids)
        return ids

    ## Abolfazl wrote
    def pclasses(self):
        # print('!!!!!!!!!! pclasses')
        # print(self.t)
        # print(len(self.t))
        # print(self.t[1].shape)

        ##### to transform the list to tensor
        t = torch.zeros((len(self.t), len(self.t[0]), len(self.t[0])))
        for i in range(len(self.t)):
            t[i] = self.t[i]
        # print(t.shape)
        return t

    def t_background(self, b=None):
        if b is not None:
            return self.t_background()[b]
        else:
            return self.pclasses() == 0

    def t_foreground(self, b=None):
        if b is not None:
            return self.t_foreground()[b]
        else:
            return self.pclasses() != 0

    def max_instance(self, b):
        return max(self.unique_ids(b))

    def n_instances(self, b):
        """number of truth instances in image"""
        return len(self.instance_ids(b))

    def height_width(self):
        """height and width of an image"""
        return self.t[0].shape[:]

    def n_pixels(self):
        """number of pixels in an image"""
        H, W = self.height_width()
        return H*W

    def n_images(self):
        """number of images in batch"""
        # print('func number of images in batch')
        # print(len(self.t))
        return len(self.t)

    def l22n(self, b=..., i=..., split=False, *, embedding_map=None, centroid=None, psigma=None):
        """l2 square normalized pixel and instance embdding for every pixel"""
        if embedding_map is None:   embedding_map = self.embedding(b)
        if centroid is None:        centroid = self.inst_embedding(b,i)
        if psigma is None:          psigma = self.inst_psigma(b,i)
        if split: return [
            self.l22n(embedding_map=embedding_map, centroid=centroid, psigma=psigma)
            for embedding_map, centroid, psigma in
            zip(self.split(embedding_map), self.split(centroid), self.split(psigma))
            ]
        dist = (embedding_map - centroid[:,None,None])**2
        dist = dist * psigma[:,None,None]
        return dist.sum(dim=0,keepdim=True)

    def gaussian(self, b=..., i=..., split=False, *, l22n=None, **kwargs):
        """gaussian between pixel and instance embeddings for every pixel"""
        if split:
            spatial_l22n, assoc_l22n = self.l22n(b,i, True, **kwargs)
            return self.gaussian(l22n=spatial_l22n), self.gaussian(l22n=assoc_l22n)
        if l22n is None:    l22n = self.l22n(b,i, **kwargs)
        return torch.exp(-0.5*l22n)

    def masked_gaussian(self, b, i, **kwargs):
        """gaussian between pixel and instance embeddings for every pixel inside of instance"""
        return self._masked(self.gaussian(b,i, **kwargs), self.inst_mask(b,i))

    def inst_mask(self, b, i):
        """truth mask of instance"""
        return self.instances(b) == i

    def mask_float(self, b, i):
        """truth mask of instance, as a float (0.0 and 1.0)"""
        return self.inst_mask(b,i).double()

    def inst_embedding(self, b, i=..., **kwargs):
        """predicted embedding for instance"""
        return self.embedding(b,i, **kwargs).mean(dim=1)

    def inst_ppsigma(self, b, i=..., **kwargs):
        return self.ppsigma(b,i, **kwargs).mean(dim=1)

    def inst_psigma(self, b, i):
        return self._psigma(self.inst_ppsigma(b,i))

    def _psigma(self, ppsigma):
        """Convert pre-pseudo-sigma to pseudo-sigma"""
        return torch.exp(10*ppsigma)

    def _sigma(self, psigma):
        """Compute true sigma from pseudo-sigma"""
        raise NotImplementedError

    def ppsigma(self, b=None, i=..., *, mask=None):
        """predicted pre-pseudo-sigma on every pixel of batch/image/instance"""
        if i is not ...:
            return self._masked(self.ppsigma(b), self.inst_mask(b,i))
        elif mask is not None:
            return self._masked(self.ppsigma(b), mask)
        elif b is not None:
            return self.ppsigma()[b]
        else:
            # ppsigma = self.fetches.y.sigma
            ppsigma = self.x_sigma
            return ppsigma

    def seed(self, b=None,i=None, **kwargs):
        """predicted seed on every pixel of batch/image/instance"""
        if i is not None:
            return self._masked(self.seed(b, **kwargs), self.inst_mask(b,i))
        elif b is not None:
            return self.seed(**kwargs)[b]
        else:
            # return self.fetches.y.seed.sigmoid()
            return self.x_seed.sigmoid()

    # def t_inst_class(self, b, *, mask):
    #     """predicted class from a mask"""
    #     classes = self._masked(self.t_classes(b), mask).squeeze(0)
    #     classes = F.one_hot(classes, self.exp.hparams.classes).float()
    #     classes = classes.mean(dim=0)
    #     clss = classes.argmax(dim=0)
    #     return clss

    # def y_inst_class(self, b, *, mask):
    #     """predicted class from a mask"""
    #     classes = self._masked(self.y_classes(b), mask)
    #     classes = classes.mean(dim=1)
    #     clss = classes.argmax(dim=0)
    #     return clss

    # def y_classes(self, b=None, logits=False):
    #     """predicted classes over image/batch as [N,C,H,W] tensor. Applies softmax if logits=False (default)"""
    #     if b is not None:
    #         return self.y_classes(logits=logits)[b]
    #     if not logits:
    #         return F.softmax(self.y_classes(logits=True), dim=1)
    #     else:
    #         return self.fetches.y.classes

    # def classes_loss(self, b=None):
    #     """classification loss"""
    #     if b is not None:
    #         logits = self.y_classes(b,logits=True)[None]
    #         target = self.t_classes(b)
    #         loss = self.exp.cross_entropy(logits, target)
    #         return loss.mean()  # across pixels
    #     else:
    #         loss = []
    #         for b in range(self.n_images()):
    #             loss.append(self.classes_loss(b))
    #         return torch.stack(loss).mean()  # across images

    # def t_classes(self, b=None):
    #     """truth labels over image/batch [H,W]/[N,H,W], value 0 being the background"""
    #     if b is not None:
    #         return self.t_classes()[b]
    #     else:
    #         classes = self.raw_pclasses().long()
    #         return classes

    # @WithCache.no_cache
    def split(self, tensor):
        """split a tensor into spatial and associative parts"""
        channels = self.embd_channels()
        
        slices = []
        start = 0
        for ch in channels:
            slices.append(slice(start, start+ch))
            start = start+ch

        if len(tensor.shape) >= 4:
            assert tensor.shape[1] == start
            return [tensor[:,slc] for slc in slices]
        assert tensor.shape[0] == start
        return [tensor[slc] for slc in slices]

    def spatial_offset(self):
        # spatial = getattr(self.fetches.y, "spatial", None)
        spatial = self.x_vec
        # print("spatial_shape")
        # print(spatial.shape)
        if spatial is not None:
            spatial = spatial.tanh()
        return spatial

    def spatial_embedding(self):
        spatial = self.spatial_offset()
        if spatial is not None:
            N, _, _, C, H, W = spatial.shape
            meshgrid = self._meshgrid()[:,:H,:W]
            spatial = spatial.view(N,C//2,2,H,W) + meshgrid[None,None,:,:,:]
            spatial = spatial.view(N,C,H,W)
        return spatial

    def associative_embedding(self):
        # assoc = getattr(self.fetches.y, "associative", None)
        assoc = self.x_vec
        N, _, _, C, H, W = assoc.shape
        assoc = assoc.view(N,C,H,W)
        if assoc is not None:
            assoc = assoc.tanh()
        return assoc

    # def pointer(self):
    #     return self.fetches.y.pointer.tanh()
    # def pointer_target(self):
    #     pointer = self.pointer()
    #     N, C, H, W = pointer.shape
    #     meshgrid = self.exp._meshgrid[:,:H,:W]
    #     target = pointer.view(N,C//2,2,H,W) + meshgrid[None,None,:,:,:]
    #     target = target.view(N,C,H,W)
    #     return target

    def embd_channels(self):
        return self.embedding()._embd_channels

    # @WithCache.no_cache
    # def resample(self, tensor, detach_pointer=False):
    #     H, W = self.height_width()
    #     # grid_sample input is [N,H,W,C], OK
    #     # grid must be [N,H,W,2(x,y)], with normalized coordinates, is [N,2,H,W], unnormalized
    #     grid = self.pointer_target()*1024
    #     grid = torch.stack([2*grid[:,1]/W-1, 2*grid[:,0]/H-1], dim=-1)
    #     if detach_pointer:
    #         grid = grid.detach()
    
    #     resampled = F.grid_sample(tensor, grid, "bilinear", padding_mode="border", align_corners=True)
    #     return resampled

    def embedding(self, b=None,i=..., *, mask=None, **kwargs):
        """predicted embedding on every pixel of batch/image/instance"""
        if i is not ...:
            return self._masked(self.embedding(b, **kwargs), self.inst_mask(b,i))
        elif mask is not None:
            return self._masked(self.embedding(b, **kwargs), mask)
        elif b is not None:
            return self.embedding(**kwargs)[b]
        else:
            embeddings = [self.spatial_embedding(), self.associative_embedding()]
            channels = [embd.shape[1] if embd is not None else 0
                        for embd in embeddings]
            embeddings = [embd for embd in embeddings if embd is not None]

            tensor = torch.cat(embeddings, dim=1)
            tensor._embd_channels = channels
            return tensor

    def raw_instances(self):
        # print('raaw_instances')
        # print(len(self.t))
        # print(self.t[0].shape)
        # print(self.t.shape)
        return self.t

    def instances(self, b=None):
        """truth instance id on every pixel of batch/image"""
        if b is not None:
            return self.instances()[b]
        else:
            instances = self.raw_instances()
            return instances

    def sigma_loss(self, b=None, i=None):
        """sigma loss supervising pixels' sigmas towards instance's mean sigma"""
        if i is not None:
            loss = (self.ppsigma(b,i) - self.inst_ppsigma(b,i)[:,None].detach())**2
            loss = loss.sum(dim=0)  # across channels
            loss = loss.mean()      # across space
            return loss
        elif b is not None:
            loss = []
            for i in self.instance_ids(b):
                loss.append(self.sigma_loss(b, i))
            # loss = loss or (self.exp._zero,)   # handle if no instance
            loss = loss or (torch.zeros([], dtype=torch.double, device=self.device),)   # handle if no instance
            return torch.stack(loss).mean()     # across instances
        else:
            loss = []
            for b in range(self.n_images()):
                loss.append(self.sigma_loss(b))
            return torch.stack(loss).mean()  # across images

    def seed_loss(self, b=None, i=None):
        """seed loss regressing pixel confidence in its output embedding"""
        if i is not None:
            if i == 0:  # background
                loss = (self._masked(self.seed(b), self.t_background(b)) - 0)**2
            else:       # instance
                loss = (self.seed(b,i) - self.masked_gaussian(b,i).detach())**2
            loss = loss.squeeze(dim=0)  # only handles one map
            loss = loss.sum()           # across space
            return loss
        elif b is not None:
            loss = []
            for i in self.unique_ids(b):
                loss.append(self.seed_loss(b,i))
            return torch.stack(loss).sum()/self.n_pixels()  # across instances
        else:
            loss = []
            for b in range(self.n_images()):
                loss.append(self.seed_loss(b))
            return torch.stack(loss).mean()  # across images

    def embedding_loss(self, b=None, i=None):
        """fused push-pull loss using a classification loss on top of distance kernel"""
        if i is not None:
            logits = 2*self.gaussian(b,i)-1
            truth = self.mask_float(b,i)
            # print('func embedding loss')
            # print(logits.type)
            # print(truth.type)
            loss = lovasz_hinge_flat(logits.view(-1), truth.view(-1))
            return loss.mean()      # across pixels
        elif b is not None:
            loss = []
            for i in self.instance_ids(b):
                loss.append(self.embedding_loss(b, i))
            # loss = loss or (self.exp._zero,)   # handle if no instance
            loss = loss or (torch.zeros([], dtype=torch.double, device=self.device),)   # handle if no instance
            # print('embedding loss (b)')
            # print(loss)
            # print(len(loss))
            return torch.stack(loss).mean()     # across instances
        else:
            loss = []
            for b in range(self.n_images()):
                loss.append(self.embedding_loss(b))
            return torch.stack(loss).mean()  # across images

    def loss(self):
        loss1 = self.embedding_loss()
        loss2 = self.seed_loss()
        loss3 = self.sigma_loss()
        # print('seg losses')
        # print(loss1)
        # print(loss2)
        # print(loss3)
        # return [self.embedding_loss()]+ self.seed_loss()+ self.sigma_loss()
        return [loss1]+ [loss2] + [loss3]

    
    def forward(self, *args):
        x, t = args

        self.x = [xx.double() for xx in x]
        self.t = [tt.double() for tt in t]

        # for path_check in range(12):
        #     filename = 'self_target_{}.pt'.format(path_check)
        #     if not path.exists(filename):
        #         torch.save(self.t, filename)
        #         # print('filename is')
        #         # print(filename)
        #         break
        # torch.save(self.t, 'last.pt')
        # print('loss')
        # print('seg instances')
        # print(len(self.x))
        # print(self.x[0].shape)
        # print(len(self.t))
        # for iiii in range(len(self.t)):
        #     print(self.t[iiii].shape)
        # print(self.t.shape)

        self.x_seed, self.x_vec, self.x_sigma = self.x
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!SEG OUTPUT')
        loss_out = self.loss()
        # for path_check in range(12):
        #     filename = 'self_target2_{}.pt'.format(path_check)
        #     if not path.exists(filename):
        #         torch.save(self.t, filename)
        #         print('filename is')
        #         print(filename)
        #         break
        # print(loss_out)
        return loss_out


def cli(parser):
    group = parser.add_argument_group('losses')
    group.add_argument('--lambdas', default=None, type=float, nargs='+',
                       help='prefactor for head losses')
    group.add_argument('--r-smooth', type=float, default=SmoothL1Loss.r_smooth,
                       help='r_{smooth} for SmoothL1 regressions')
    group.add_argument('--regression-loss', default='laplace',
                       choices=['smoothl1', 'smootherl1', 'l1', 'laplace'],
                       help='type of regression loss')
    group.add_argument('--background-weight', default=CompositeLoss.background_weight, type=float,
                       help='BCE weight where ground truth is background')
    group.add_argument('--focal-gamma', default=CompositeLoss.focal_gamma, type=float,
                       help='when > 0.0, use focal loss with the given gamma')
    group.add_argument('--margin-loss', default=False, action='store_true',
                       help='[experimental]')
    group.add_argument('--auto-tune-mtl', default=False, action='store_true',
                       help='use Kendall\'s prescription for adjusting the multitask weight')
    assert MultiHeadLoss.task_sparsity_weight == MultiHeadLossAutoTune.task_sparsity_weight
    group.add_argument('--task-sparsity-weight',
                       default=MultiHeadLoss.task_sparsity_weight, type=float,
                       help='[experimental]')


def configure(args):
    # apply for CompositeLoss
    CompositeLoss.background_weight = args.background_weight
    CompositeLoss.focal_gamma = args.focal_gamma
    CompositeLoss.margin = args.margin_loss

    # MultiHeadLoss
    MultiHeadLoss.task_sparsity_weight = args.task_sparsity_weight
    MultiHeadLossAutoTune.task_sparsity_weight = args.task_sparsity_weight

    # SmoothL1
    SmoothL1Loss.r_smooth = args.r_smooth


def factory_from_args(args, head_nets):
    return factory(
        head_nets,
        args.lambdas,
        reg_loss_name=args.regression_loss,
        device=args.device,
        auto_tune_mtl=args.auto_tune_mtl,
    )


# pylint: disable=too-many-branches
def factory(head_nets, lambdas, *,
            reg_loss_name=None, device=None,
            auto_tune_mtl=False):
    if isinstance(head_nets[0], (list, tuple)):
        return [factory(hn, lam,
                        reg_loss_name=reg_loss_name,
                        device=device)
                for hn, lam in zip(head_nets, lambdas)]

    if reg_loss_name == 'smoothl1':
        reg_loss = SmoothL1Loss()
    elif reg_loss_name == 'l1':
        reg_loss = l1_loss
    elif reg_loss_name == 'laplace':
        reg_loss = laplace_loss
    elif reg_loss_name is None:
        reg_loss = laplace_loss
    else:
        raise Exception('unknown regression loss type {}'.format(reg_loss_name))

    sparse_task_parameters = None
    if MultiHeadLoss.task_sparsity_weight:
        sparse_task_parameters = []
        for head_net in head_nets:
            if getattr(head_net, 'sparse_task_parameters', None) is not None:
                sparse_task_parameters += head_net.sparse_task_parameters
            elif isinstance(head_net, heads.CompositeFieldFused):
                sparse_task_parameters.append(head_net.conv.weight)
            else:
                raise Exception('unknown l1 parameters for given head: {} ({})'
                                ''.format(head_net.meta.name, type(head_net)))

    # losses = [CompositeLoss(head_net, reg_loss) for head_net in head_nets]
    ### AMA
    losses = [CompositeLoss(head_nets[0], reg_loss)]
    losses.append(SegmantationLoss(head_nets[1],device))
    if auto_tune_mtl:
        loss = MultiHeadLossAutoTune(losses, lambdas,
                                     sparse_task_parameters=sparse_task_parameters)
    else:
        loss = MultiHeadLoss(losses, lambdas)

    if device is not None:
        loss = loss.to(device)

    return loss
