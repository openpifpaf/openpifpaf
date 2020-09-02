class DataModule():
    """Interface for custom data.

    Subclass this class to incorporate your custom dataset.
    On the command line, use this module with its full Python import path.

    Overwrite cli() and configure() to make your data module configureable
    from the command line.
    """
    batch_size = 8
    loader_workers = None

    head_metas = None  # make instance(!) variable (not class variable) in derived classes

    @classmethod
    def cli(cls, parser):
        pass

    @classmethod
    def configure(cls, args):
        pass

    def train_loader(self):
        raise NotImplementedError

    def val_loader(self):
        raise NotImplementedError

    def test_loader(self):
        raise NotImplementedError
