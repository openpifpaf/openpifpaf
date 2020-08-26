class DataModule():
    """Interface for custom data.

    Subclass this class to incorporate your custom dataset.
    On the command line, use this module with its full Python import path.

    Overwrite cli() and configure() to make your data module configureable
    from the command line.
    """
    description = ''

    @classmethod
    def cli(cls, parser):
        pass

    @classmethod
    def configure(cls, args):
        pass

    def head_metas(self):
        raise NotImplementedError

    def train_loader(self, target_transforms):
        raise NotImplementedError

    def val_loader(self, target_transforms):
        raise NotImplementedError

    def test_loader(self):
        raise NotImplementedError
