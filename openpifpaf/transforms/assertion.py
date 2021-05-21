from .preprocess import Preprocess


class Assert(Preprocess):
    """Inspect (and assert) on current image, anns, meta."""
    def __init__(self, function, message=None):
        self.function = function
        self.message = message

    def __call__(self, *args):
        assert self.function(*args), self.message

        return args
