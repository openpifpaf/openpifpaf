import logging
import torch

LOG = logging.getLogger(__name__)


def cli(parser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum, beta1 in Adam')
    group.add_argument('--beta2', type=float, default=0.999,
                       help='beta2 for Adam/AMSGrad')
    group.add_argument('--adam-eps', type=float, default=1e-6,
                       help='eps value for Adam/AMSGrad')
    group.add_argument('--no-nesterov', dest='nesterov', default=True, action='store_false',
                       help='do not use Nesterov momentum for SGD update')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='SGD/Adam/AMSGrad weight decay')
    group.add_argument('--adam', action='store_true',
                       help='use Adam optimizer')
    group.add_argument('--amsgrad', action='store_true',
                       help='use Adam optimizer with AMSGrad option')

    group_s = parser.add_argument_group('learning rate scheduler')
    group_s.add_argument('--lr', type=float, default=1e-3,
                         help='learning rate')
    group_s.add_argument('--lr-decay', default=[], nargs='+', type=float,
                         help='epochs at which to decay the learning rate')
    group_s.add_argument('--lr-decay-factor', default=0.1, type=float,
                         help='learning rate decay factor')
    group_s.add_argument('--lr-decay-epochs', default=1.0, type=float,
                         help='learning rate decay duration in epochs')
    group_s.add_argument('--lr-warm-up-start-epoch', default=0, type=float,
                         help='starting epoch for warm-up')
    group_s.add_argument('--lr-warm-up-epochs', default=1, type=float,
                         help='number of epochs at the beginning with lower learning rate')
    group_s.add_argument('--lr-warm-up-factor', default=0.001, type=float,
                         help='learning pre-factor during warm-up')


class LearningRateLambda(object):
    def __init__(self, decay_schedule, *,
                 decay_factor=0.1,
                 decay_epochs=1.0,
                 warm_up_start_epoch=0,
                 warm_up_epochs=2.0,
                 warm_up_factor=0.01):
        self.decay_schedule = decay_schedule
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warm_up_start_epoch = warm_up_start_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor

    def __call__(self, step_i):
        lambda_ = 1.0

        if step_i <= self.warm_up_start_epoch:
            lambda_ *= self.warm_up_factor

        if self.warm_up_start_epoch < step_i < self.warm_up_start_epoch + self.warm_up_epochs:
            lambda_ *= self.warm_up_factor**(
                1.0 - (step_i - self.warm_up_start_epoch) / self.warm_up_epochs
            )

        for d in self.decay_schedule:
            if step_i >= d + self.decay_epochs:
                lambda_ *= self.decay_factor
            elif step_i > d:
                lambda_ *= self.decay_factor**(
                    (step_i - d) / self.decay_epochs
                )

        return lambda_


def factory_optimizer(args, parameters):
    if args.amsgrad:
        args.adam = True

    if args.adam:
        LOG.info('Adam optimizer')
        optimizer = torch.optim.Adam(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, betas=(args.momentum, args.beta2),
            weight_decay=args.weight_decay, eps=args.adam_eps, amsgrad=args.amsgrad)
    else:
        LOG.info('SGD optimizer')
        optimizer = torch.optim.SGD(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=args.nesterov)

    return optimizer


def factory_lrscheduler(args, optimizer, training_batches_per_epoch):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [
            LearningRateLambda(
                [s * training_batches_per_epoch for s in args.lr_decay],
                decay_factor=args.lr_decay_factor,
                decay_epochs=args.lr_decay_epochs * training_batches_per_epoch,
                warm_up_start_epoch=args.lr_warm_up_start_epoch * training_batches_per_epoch,
                warm_up_epochs=args.lr_warm_up_epochs * training_batches_per_epoch,
                warm_up_factor=args.lr_warm_up_factor,
            ),
        ],
    )
