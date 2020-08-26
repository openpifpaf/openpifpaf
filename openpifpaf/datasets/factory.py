DATAMODULES = set()


def train_cli(parser):
    for dm in DATAMODULES:
        dm.cli(parser)

    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--dataset', default='cocokp')


def train_configure(args):
    for dm in DATAMODULES:
        dm.configure(args)


def datamodule_factory(dataset):
    dataset_lower = dataset.lower()

    for dm in DATAMODULES:
        if dataset_lower == dm.__name__.lower():
            return dm()

    return None
