import logging

import numpy as np

from openpifpaf.metric.base import Base
from openpifpaf.annotation import Annotation

try:
    import scipy
except ImportError:
    scipy = None

LOG = logging.getLogger(__name__)


class MeanPixelError(Base):
    """
    Calculate mean pixel error and detection rate for a given image
    and category in an "all-vs-all setting"
    """
    predictions = []
    image_ids = []
    errors = []  # mean pixel errors
    detections = []  # detection rate
    errors_scaled = []  # mean pixel errors
    detections_scaled = []  # detection rate
    px_ref = 368  # CPM crop size in pixels

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        errors = []
        detections = []
        errors_scaled = []
        detections_scaled = []

        # Filter ground-truth
        for annotation in ground_truth:
            if not isinstance(annotation, Annotation):
                continue
            indices_gt = np.nonzero(annotation.data[:, 2] > 1.0)
            if indices_gt[0].size <= 3:
                continue
            gts = annotation.data[indices_gt, 0:2].squeeze()
            width = float(annotation.fixed_bbox[2])
            height = float(annotation.fixed_bbox[3])
            scale = np.array([self.px_ref / width, self.px_ref / height]).reshape(1, 2)

            # Evaluate each keypoint
            for idx, gt in zip(indices_gt[0], gts):
                preds = np.array([p.data[idx] for p in predictions]).reshape(-1, 3)[:, 0:2]
                if preds.size <= 0:
                    continue
                i = np.argmin(np.linalg.norm(preds - gt, axis=1))
                dist = preds[i:i + 1] - gt
                dist_scaled = dist * scale
                d = float(np.linalg.norm(dist, axis=1))
                d_scaled = float(np.linalg.norm(dist_scaled, axis=1))

                # Prediction correct if error less than 10 pixels
                if d < 10:
                    errors.append(d)
                    detections.append(1)
                else:
                    detections.append(0)
                if d_scaled < 10:
                    errors_scaled.append(d)
                    detections_scaled.append(1)
                else:
                    detections_scaled.append(0)

        # Stats for a single image
        mpe = average(errors)
        mpe_scaled = average(errors_scaled)
        det_rate = 100 * average(detections)
        det_rate_scaled = 100 * average(detections_scaled)
        LOG.info('Mean Pixel Error (scaled): %s (%s)    Det. Rate (scaled): %s (%s)',
                 str(mpe)[:4], str(mpe_scaled)[:4], str(det_rate)[:4], str(det_rate_scaled)[:4])

        # Accumulate stats
        self.errors.extend(errors)
        self.detections.extend(detections)
        self.errors_scaled.extend(errors_scaled)
        self.detections_scaled.extend(detections_scaled)

    def write_predictions(self, filename, *, additional_data=None):
        raise NotImplementedError

    def stats(self):
        mpe = average(self.errors)
        mpe_scaled = average(self.errors_scaled)
        det_rate = 100 * average(self.detections)
        det_rate_scaled = 100 * average(self.detections_scaled)
        LOG.info('Final Results: \nMean Pixel Error [scaled] : %f [%f] '
                 '\nDetection Rate [scaled]: %f [%f]',
                 mpe, mpe_scaled, det_rate, det_rate_scaled)
        data = {
            'stats': [mpe, mpe_scaled, det_rate, det_rate_scaled],
            'text_labels': ['Mean Pixel Error',
                            'Mean Pixel Error Scaled',
                            'Detection Rate [%]',
                            'Detection Rate Scaled[%]'],
        }
        return data


def hungarian_matching(gts, predictions, thresh=0.5):
    cost = np.zeros((len(gts), len(predictions)))

    for i, (dg, vg) in enumerate(gts):
        for j, pred in enumerate(predictions):
            p = np.array(pred.data)
            dp = p[:, 0:2][vg > 1.0]
            vp = p[:, 2][vg > 1.0]

            dp[vp < thresh] = -100
            dp[vp < thresh] = -100

            # measure the per-keypoint distance
            distances = np.clip(np.linalg.norm(dp - dg, axis=1), 0, 10)
            cost[i, j] = float(np.mean(distances))

    assert np.max(cost) < 11
    row, cols = scipy.optimize.linear_sum_assignment(cost)
    return row, cols, cost


def average(my_list, *, empty_value=0.0):
    """calculate mean of a list"""
    if not my_list:
        return empty_value

    return sum(my_list) / float(len(my_list))
