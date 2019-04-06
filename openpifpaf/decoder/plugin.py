from abc import ABCMeta, abstractmethod, abstractstaticmethod


class Plugin(metaclass=ABCMeta):
    @abstractstaticmethod
    def match(head_names):  # pylint: disable=unused-argument
        return False

    @abstractmethod
    def __call__(self, fields):
        pass
