class Base:
    def __init__(self):
        self.nn_time = 0.0
        self.decoder_time = 0.0

    def accumulate(self, predictions, image_meta):
        raise NotImplementedError

    def write_predictions(self, filename):
        raise NotImplementedError

    def write_stats(self, filename, *, additional_data):
        raise NotImplementedError
