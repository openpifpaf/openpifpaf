import logging
import json
import zipfile
import copy

import numpy as np
try:
    from xtcocotools.cocoeval import COCOeval
except ImportError:
    pass
from openpifpaf.metric.base import Base

try:
    import pycocotools.coco
except ImportError:
    pass

LOG = logging.getLogger(__name__)


class WholebodyMetric(Base):
    text_labels_keypoints = ['AP', 'AP0.5', 'AP0.75', 'APM', 'APL',
                             'AR', 'AR0.5', 'AR0.75', 'ARM', 'ARL']
    text_labels_bbox = ['AP', 'AP0.5', 'AP0.75', 'APS', 'APM', 'APL',
                        'ART1', 'ART10', 'AR', 'ARS', 'ARM', 'ARL']

    def __init__(self, coco, *,
                 max_per_image=20,
                 category_ids=None,
                 iou_type='keypoints',
                 small_threshold=0.0,
                 keypoint_oks_sigmas=None):
        super().__init__()

        if category_ids is None:
            category_ids = [1]

        self.max_per_image = max_per_image
        self.category_ids = category_ids
        self.iou_type = iou_type
        self.small_threshold = small_threshold
        self.keypoint_oks_sigmas = keypoint_oks_sigmas

        self.predictions = []
        self.image_ids = []
        self.eval = None
        self.coco = pycocotools.coco.COCO(coco)

        self.sigmas_body = [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                            0.062, 0.062, 0.107, 0.107, 0.087,
                            0.087, 0.089, 0.089]
        self.sigmas_foot = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
        self.sigmas_face = [0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031, 0.025, 0.020, 0.023,
                            0.029, 0.032, 0.037, 0.038, 0.043,
                            0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011,
                            0.013, 0.015, 0.009, 0.007, 0.007,
                            0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
                            0.009, 0.007, 0.013, 0.008, 0.011,
                            0.012, 0.010, 0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010,
                            0.008, 0.009, 0.009, 0.009, 0.007,
                            0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01, 0.008]
        self.sigmas_lefthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
                                0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
                                0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]
        self.sigmas_righthand = [0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
                                 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
                                 0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031]

        self.sigmas_wholebody = self.sigmas_body + self.sigmas_foot + self.sigmas_face + \
            self.sigmas_lefthand + self.sigmas_righthand

        if self.iou_type == 'keypoints':
            self.text_labels = self.text_labels_keypoints
        elif self.iou_type == 'bbox':
            self.text_labels = self.text_labels_bbox
        else:
            LOG.warning('Unknown iou type "%s". Specify text_labels yourself.', self.iou_type)

        LOG.debug('max = %d, category ids = %s, iou_type = %s',
                  self.max_per_image, self.category_ids, self.iou_type)

    def _stats(self, predictions=None, image_ids=None):
        # from pycocotools.cocoeval import COCOeval
        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids

        coco_eval = self.coco.loadRes(predictions)

        for count in coco_eval.anns:
            ann_orig = copy.deepcopy(coco_eval.anns[count])
            coco_eval.anns[count]["lefthand_kpts"] = ann_orig["keypoints"][91 * 3:112 * 3]
            coco_eval.anns[count]["righthand_kpts"] = ann_orig["keypoints"][112 * 3:133 * 3]
            coco_eval.anns[count]["face_kpts"] = ann_orig["keypoints"][23 * 3:91 * 3]
            coco_eval.anns[count]["foot_kpts"] = ann_orig["keypoints"][17 * 3:23 * 3]
            coco_eval.anns[count]["keypoints"] = ann_orig["keypoints"][0:17 * 3]

        coco_eval.anno_file = coco_eval.anns  # xtpycocotools style
        self.coco.anno_file = self.coco.anns
        eval_body = COCOeval(self.coco, coco_eval, 'keypoints_body', np.array(self.sigmas_body),
                             use_area=True)
        eval_body.evaluate()
        eval_body.accumulate()
        eval_body.summarize()
        eval_foot = COCOeval(self.coco, coco_eval, 'keypoints_foot', np.array(self.sigmas_foot),
                             use_area=True)
        eval_foot.evaluate()
        eval_foot.accumulate()
        eval_foot.summarize()
        eval_face = COCOeval(self.coco, coco_eval, 'keypoints_face', np.array(self.sigmas_face),
                             use_area=True)
        eval_face.evaluate()
        eval_face.accumulate()
        eval_face.summarize()
        eval_lh = COCOeval(self.coco, coco_eval, 'keypoints_lefthand',
                           np.array(self.sigmas_lefthand), use_area=True)
        eval_lh.evaluate()
        eval_lh.accumulate()
        eval_lh.summarize()
        eval_rh = COCOeval(self.coco, coco_eval, 'keypoints_righthand',
                           np.array(self.sigmas_righthand), use_area=True)
        eval_rh.evaluate()
        eval_rh.accumulate()
        eval_rh.summarize()
        eval_wb = COCOeval(self.coco, coco_eval, 'keypoints_wholebody',
                           np.array(self.sigmas_wholebody), use_area=True)
        eval_wb.evaluate()
        eval_wb.accumulate()
        eval_wb.summarize()
        return eval_wb.stats

    # pylint: disable=unused-argument
    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        image_id = int(image_meta['image_id'])
        self.image_ids.append(image_id)

        if self.small_threshold:
            predictions = [pred for pred in predictions
                           if pred.scale(v_th=0.01) >= self.small_threshold]
        if len(predictions) > self.max_per_image:
            predictions = predictions[:self.max_per_image]

        image_annotations = []
        for pred in predictions:
            pred_data = pred.json_data()
            pred_data['image_id'] = image_id
            pred_data = {
                k: v for k, v in pred_data.items()
                if k in ('category_id', 'score', 'keypoints', 'bbox', 'image_id')
            }
            image_annotations.append(pred_data)

        # force at least one annotation per image (for pycocotools)
        if not image_annotations:
            n_keypoints = 133
            image_annotations.append({
                'image_id': image_id,
                'category_id': 1,
                'keypoints': np.zeros((n_keypoints * 3,)).tolist(),
                'bbox': [0, 0, 1, 1],
                'score': 0.001,
            })

        if LOG.getEffectiveLevel() == logging.DEBUG:
            self._stats(image_annotations, [image_id])
            LOG.debug(image_meta)

        self.predictions += image_annotations

    def write_predictions(self, filename, *, additional_data=None):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        predictions_wb = copy.deepcopy(predictions)
        for ann in predictions_wb:
            ann_orig = copy.deepcopy(ann)
            ann["lefthand_kpts"] = ann_orig["keypoints"][91 * 3:112 * 3]
            ann["righthand_kpts"] = ann_orig["keypoints"][112 * 3:133 * 3]
            ann["face_kpts"] = ann_orig["keypoints"][23 * 3:91 * 3]
            ann["foot_kpts"] = ann_orig["keypoints"][17 * 3:23 * 3]
            ann["keypoints"] = ann_orig["keypoints"][0:17 * 3]
        with open(filename + '.pred_wb.json', 'w') as f:
            json.dump(predictions_wb, f)
        LOG.info('wrote %s.pred_wb.json', filename)
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)

    def stats(self):
        data = {
            'stats': self._stats().tolist(),
            'text_labels': self.text_labels,
        }

        return data
