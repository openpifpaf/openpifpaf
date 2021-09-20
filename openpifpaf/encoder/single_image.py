class SingleImage:
    """Helper class for encoders on datasets with image pairs."""

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __call__(self, images, anns, metas):
        return self.wrapped(images[0], anns[0], metas[0])

    def __repr__(self):
        return __class__.__module__ + '.' + __class__.__name__ + '(' + repr(self.wrapped) + ')'

    @property
    def meta(self):
        return self.wrapped.meta
