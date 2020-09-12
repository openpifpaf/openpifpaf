class Base:
    def __init__(self):
        #: accumulated time spent processing the neural network part
        self.nn_time = 0.0

        #: accumulated time spent in the decoder
        self.decoder_time = 0.0

    def accumulate(self, predictions, image_meta):
        """For every image, accumulate that image's predictions into this metric.

        :param predictions: List of predictions for one image.
        :param image_meta: Meta dictionary for this image as returned by the data loader.
        """
        raise NotImplementedError

    def stats(self):
        """Return a dictionary of summary statistics."""
        raise NotImplementedError

    def write_predictions(self, filename):
        """Write predictions to a file.

        This is used to produce a metric-compatible output of predictions.
        It is used for test challenge submissions where a remote server
        holds the private test set.

        :param filename: Output filename of prediction file.
        """
        raise NotImplementedError
