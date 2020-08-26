from .cifar10 import Cifar10
from .cocodet import CocoDet
from .cocokp import CocoKp
from .module import DataModule

DATAMODULES = {Cifar10, CocoDet, CocoKp}


def factory(dataset):
    dataset_lower = dataset.lower()

    for dm in DATAMODULES:
        if dataset_lower == dm.__name__.lower():
            return dm()

    return None


def cli(parser):
    group = parser.add_argument_group('generic data module parameters')
    group.add_argument('--dataset')
    group.add_argument('--loader-workers',
                        default=DataModule.loader_workers, type=int,
                        help='number of workers for data loading')
    group.add_argument('--batch-size',
                        default=DataModule.batch_size, type=int,
                        help='batch size')

    for dm in DATAMODULES:
        dm.cli(parser)


def configure(args):
    DataModule.loader_workers = args.loader_workers
    DataModule.batch_size = args.batch_size

    if DataModule.loader_workers is None:
        DataModule.loader_workers = DataModule.batch_size

    for dm in DATAMODULES:
        dm.configure(args)
