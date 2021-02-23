import logging

import torch

LOG = logging.getLogger(__name__)


class MultiHeadLoss(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas):
        super().__init__()

        if not lambdas:
            lambdas = [1.0 for l in losses for _ in l.field_names]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = torch.nn.ModuleList(losses)
        self.lambdas = lambdas

        self.field_names = [n for l in self.losses for n in l.field_names]
        assert len(self.field_names) == len(self.lambdas)
        LOG.info('multihead loss: %s, %s', self.field_names, self.lambdas)

    def forward(self, head_fields, head_targets):  # pylint: disable=arguments-differ
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        assert self.task_sparsity_weight == 0.0  # TODO implement
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        assert len(self.lambdas) == len(flat_head_losses)
        loss_values = [lam * l
                       for lam, l in zip(self.lambdas, flat_head_losses)
                       if l is not None]
        total_loss = sum(loss_values) if loss_values else None

        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneKendall(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None, tune=None):
        """Auto-tuning multi-head loss.

        Uses idea from "Multi-Task Learning Using Uncertainty to Weigh Losses
        for Scene Geometry and Semantics" by Kendall, Gal and Cipolla.

        Individual losses must not be negative for Kendall's prescription.

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
        self.tune = tune

        self.log_sigmas = torch.nn.Parameter(
            torch.zeros((len(lambdas),), dtype=torch.float64),
            requires_grad=True,
        )

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.log_sigmas)

        if self.tune is None:
            def tune_from_name(name):
                if '.vec' in name:
                    return 'none'
                if '.scale' in name:
                    return 'laplace'
                return 'gauss'
            self.tune = [
                tune_from_name(n)
                for l in self.losses for n in l.field_names
            ]
        LOG.info('tune config: %s', self.tune)

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
        constrained_log_sigmas = 3.0 * torch.tanh(self.log_sigmas / 3.0)

        def tuned_loss(tune, log_sigma, loss):
            if tune == 'none':
                return loss
            if tune == 'laplace':
                # this is the negative ln of a Laplace with
                # ln(2) = 0.694
                return 0.694 + log_sigma + loss * torch.exp(-log_sigma)
            if tune == 'gauss':
                # this is the negative ln of a Gaussian with
                # ln(sqrt(2pi)) = 0.919
                return 0.919 + log_sigma + loss * 0.5 * torch.exp(-2.0 * log_sigma)
            raise Exception('unknown tune: {}'.format(tune))

        loss_values = [
            lam * tuned_loss(t, log_sigma, l)
            for lam, t, log_sigma, l in zip(
                self.lambdas, self.tune, constrained_log_sigmas, flat_head_losses)
            if l is not None
        ]
        total_loss = sum(loss_values) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses


class MultiHeadLossAutoTuneVariance(torch.nn.Module):
    task_sparsity_weight = 0.0

    def __init__(self, losses, lambdas, *, sparse_task_parameters=None):
        """Auto-tuning multi-head loss based on loss-variance.

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

        self.epsilons = torch.ones((len(lambdas),), dtype=torch.float64)
        # choose a prime number for the buffer length:
        # for multiple tasks, prevents that always the same buffer_index is
        # skipped which would mean that some nan values will remain forever
        self.buffer = torch.full((len(lambdas), 53), float('nan'), dtype=torch.float64)
        self.buffer_index = -1

        self.field_names = [n for l in self.losses for n in l.field_names]
        LOG.info('multihead loss with autotune: %s', self.field_names)
        assert len(self.field_names) == len(self.lambdas)
        assert len(self.field_names) == len(self.epsilons)

    def batch_meta(self):
        return {'mtl_sigmas': [round(float(s), 3) for s in self.epsilons]}

    def forward(self, *args):
        head_fields, head_targets = args
        LOG.debug('losses = %d, fields = %d, targets = %d',
                  len(self.losses), len(head_fields), len(head_targets))
        assert len(self.losses) == len(head_fields)
        assert len(self.losses) <= len(head_targets)
        flat_head_losses = [ll
                            for l, f, t in zip(self.losses, head_fields, head_targets)
                            for ll in l(f, t)]

        self.buffer_index = (self.buffer_index + 1) % self.buffer.shape[1]
        for i, ll in enumerate(flat_head_losses):
            if not hasattr(ll, 'data'):  # e.g. when ll is None
                continue
            self.buffer[i, self.buffer_index] = ll.data
        self.epsilons = torch.sqrt(
            torch.mean(self.buffer**2, dim=1)
            - torch.sum(self.buffer, dim=1)**2 / self.buffer.shape[1]**2
        )
        self.epsilons[torch.isnan(self.epsilons)] = 10.0
        self.epsilons = self.epsilons.clamp(0.01, 100.0)

        # normalize sum 1/eps to the number of sub-losses
        LOG.debug('eps before norm: %s', self.epsilons)
        self.epsilons = self.epsilons * torch.sum(1.0 / self.epsilons) / self.epsilons.shape[0]
        LOG.debug('eps after norm: %s', self.epsilons)

        assert len(self.lambdas) == len(flat_head_losses)
        assert len(self.epsilons) == len(flat_head_losses)
        loss_values = [
            lam * l / eps
            for lam, eps, l in zip(self.lambdas, self.epsilons, flat_head_losses)
            if l is not None
        ]
        total_loss = sum(loss_values) if loss_values else None

        if self.task_sparsity_weight and self.sparse_task_parameters is not None:
            head_sparsity_loss = sum(
                # torch.norm(param, p=1)
                param.abs().max(dim=1)[0].clamp(min=1e-6).sum()
                for param in self.sparse_task_parameters
            )
            LOG.debug('l1 head sparsity loss = %f (total = %f)', head_sparsity_loss, total_loss)
            total_loss = total_loss + self.task_sparsity_weight * head_sparsity_loss

        return total_loss, flat_head_losses
