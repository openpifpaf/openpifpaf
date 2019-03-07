import torch


def cli(parser):
    group = parser.add_argument_group('optimizer')
    group.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    group.add_argument('--no-nesterov', default=True, dest='nesterov', action='store_false',
                       help='do not use Nesterov momentum for SGD update')
    group.add_argument('--weight-decay', type=float, default=0.0,
                       help='SGD weight decay')
    group.add_argument('--adam', action='store_true',
                       help='use Adam optimizer')
    group.add_argument('--amsgrad', action='store_true',
                       help='use Adam optimizer with amsgrad option')

    group_s = parser.add_argument_group('learning rate scheduler')
    group_s.add_argument('--lr', type=float, default=1e-3,
                         help='learning rate')
    group_s.add_argument('--lr-decay', default=[], nargs='+', type=int,
                         help='epochs at which to decay the learning rate')
    group_s.add_argument('--lr-burn-in-epochs', default=1, type=int,
                         help='number of epochs at the beginning with lower learning rate')
    group_s.add_argument('--lr-burn-in-factor', default=0.1, type=float,
                         help='learning pre-factor during burn-in')
    group_s.add_argument('--lr-gamma', default=0.1, type=float,
                         help='lr decay factor')


def factory(args, parameters):
    if args.amsgrad:
        print('Adam optimizer with amsgrad')
        optimizer = torch.optim.Adam(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, eps=1e-4)
    elif args.adam:
        print('Adam optimizer')
        optimizer = torch.optim.Adam(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, weight_decay=args.weight_decay, eps=1e-4)
    else:
        print('SGD optimizer')
        optimizer = torch.optim.SGD(
            (p for p in parameters if p.requires_grad),
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
            nesterov=args.nesterov)

    def lambda_schedule(epoch):
        if epoch < args.lr_burn_in_epochs:
            return args.lr_burn_in_factor

        lambda_ = 1.0
        for d in args.lr_decay:
            if epoch >= d:
                lambda_ *= args.lr_gamma

        return lambda_

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda_schedule])
    return optimizer, scheduler
