from ..signal import Signal


class LoaderWithReset:
    """Wraps another data loader and emits a signal when a meta key changes.

    This is used for video datasets where one of the keys is the
    video sequence id. When that id changes, this loader emits a 'eval_reset'
    signal that the decoder and a RunningCache can subscribe to to reset
    their internal state.
    """

    def __init__(self, parent, key_to_monitor):
        self.parent = parent
        self.key_to_monitor = key_to_monitor

        self.previous_value = None

    def __iter__(self):
        for images, anns, metas in self.parent:
            value = metas[0][self.key_to_monitor]
            if len(metas) >= 2:
                assert all(m[self.key_to_monitor] == value for m in metas[1:])

            if value != self.previous_value:
                Signal.emit('eval_reset')
                self.previous_value = value

            yield images, anns, metas

    def __len__(self):
        return len(self.parent)
