import argparse


class Configurable:
    """Make a class configurable with CLI and by instance.

    .. warning::

        This is an experimental class.
        It is in limited use already but should not be expanded for now.

    To use this class, inherit from it in the class that you want to make
    configurable. There is nothing else to do if your class does not have
    an `__init__` method. If it does, you should take extra keyword arguments
    (`kwargs`) in the signature and pass them to the super constructor.

    Example:

    >>> class MyClass(openpifpaf.Configurable):
    ...     a = 0
    ...     def __init__(self, myclass_argument=None, **kwargs):
    ...         super().__init__(**kwargs)
    ...     def get_a(self):
    ...         return self.a
    >>> MyClass().get_a()
    0

    Instance configurability allows to overwrite a class configuration
    variable with an instance variable by passing that variable as a keyword
    into the class constructor:

    >>> MyClass(a=1).get_a()  # instance variable overwrites value locally
    1
    >>> MyClass().get_a()  # while the class variable is untouched
    0

    """
    def __init__(self, **kwargs):
        # use kwargs to set instance attributes to overwrite class attributes
        for key, value in kwargs.items():
            assert hasattr(self, key), key
            setattr(self, key, value)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Extend an ArgumentParser with the configurable parameters."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Configure the class from parsed command line arguments."""
