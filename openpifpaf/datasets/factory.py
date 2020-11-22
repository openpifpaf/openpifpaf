from .module import DataModule
from .multiloader import MultiLoader
from .multimodule import MultiDataModule

DATAMODULES = {}


def factory(dataset):
    if '-' in dataset:
        datamodules = [factory(ds) for ds in dataset.split('-')]
        return MultiDataModule(datamodules)

    if dataset not in DATAMODULES:
        raise Exception('dataset {} unknown'.format(dataset))
    return DATAMODULES[dataset]()


def cli(parser):
    group = parser.add_argument_group('generic data module parameters')
    group.add_argument('--dataset')
    group.add_argument('--loader-workers',
                       default=None, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size',
                       default=DataModule.batch_size, type=int,
                       help='batch size')
    group.add_argument('--dataset-weights', default=None, nargs='+', type=float,
                       help='n-1 weights for the datasets')

    for dm in DATAMODULES.values():
        dm.cli(parser)


def configure(args):
    DataModule.loader_workers = args.loader_workers
    DataModule.batch_size = args.batch_size

    if DataModule.loader_workers is None:
        if args.debug:
            DataModule.loader_workers = 0
        else:
            # Do not propose more than 16 loaders. More loaders use more
            # shared memory. When shared memory is exceeded, all jobs
            # on that machine crash.
            DataModule.loader_workers = min(16, DataModule.batch_size)

    MultiLoader.weights = args.dataset_weights

    for dm in DATAMODULES.values():
        dm.configure(args)
