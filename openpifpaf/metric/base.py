class Base:
    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        """For every image, accumulate that image's predictions into this metric.

        :param predictions: List of predictions for one image.
        :param image_meta: Meta dictionary for this image as returned by the data loader.
        :param ground_truth: Ground truth information as produced by the eval
            loader. Optional because some metrics (i.e. pycocotools) read
            ground truth separately.
        """
        raise NotImplementedError

    def stats(self):
        """Return a dictionary of summary statistics.

        The dictionary should be of the following form and can contain
        an arbitrary number of entries with corresponding labels:

        .. code-block::

            {
                'stats': [0.1234, 0.5134],
                'text_labels': ['AP', 'AP0.50'],
            }
        """
        raise NotImplementedError

    def write_predictions(self, filename, *, additional_data=None):
        """Write predictions to a file.

        This is used to produce a metric-compatible output of predictions.
        It is used for test challenge submissions where a remote server
        holds the private test set.

        :param filename: Output filename of prediction file.
        :param additional_data: Additional information that might be worth saving
            along with the predictions.
        """
        raise NotImplementedError
