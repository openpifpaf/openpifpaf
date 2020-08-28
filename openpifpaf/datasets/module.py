class DataModule():
    """Interface for custom data.

    Subclass this class to incorporate your custom dataset.
    On the command line, use this module with its full Python import path.

    Overwrite cli() and configure() to make your data module configureable
    from the command line.
    """
    description = ''

    batch_size = 8
    loader_workers = None

    @classmethod
    def cli(cls, parser):
        pass

    @classmethod
    def configure(cls, args):
        pass

    @classmethod
    def head_metas(cls):
        raise NotImplementedError

    def train_loader(self, base_stride):
        raise NotImplementedError

    def val_loader(self, base_stride):
        raise NotImplementedError

    def test_loader(self):
        raise NotImplementedError
