import logging

from .base import Base

LOG = logging.getLogger(__name__)


class Classification(Base):
    def __init__(self, categories):
        self.categories = ['total'] + list(categories)

        # counters: index 0 is the total
        self.gt_counts = [0 for _ in range(len(categories) + 1)]
        self.correct_counts = [0 for _ in range(len(categories) + 1)]

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # get categories
        gt_category = ground_truth[0].category_id
        if predictions:
            max_prediction = max(predictions, key=lambda p: p.score)
            pred_category = max_prediction.category_id
        else:
            pred_category = None
        LOG.debug('ground truth = %s, prediction = %s', gt_category, pred_category)

        # add to counts
        self.gt_counts[0] += 1
        self.gt_counts[gt_category] += 1
        if gt_category == pred_category:
            self.correct_counts[0] += 1
            self.correct_counts[gt_category] += 1

    def stats(self):
        return {
            'text_labels': self.categories,
            'stats': [correct / gt_count if gt_count else 0.0
                      for correct, gt_count in zip(self.correct_counts, self.gt_counts)]
        }

    def write_predictions(self, filename, *, additional_data=None):
        raise NotImplementedError
