from collections import defaultdict
import copy
import logging
import os

import torch.utils.data
from PIL import Image

import openpifpaf


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class CocoDataset(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, image_dir, ann_file, *,
                 preprocess=None, min_kp_anns=0,
                 category_ids=None,
                 annotation_filter=False):
        super().__init__()

        if category_ids is None:
            category_ids = []

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)

        self.category_ids = category_ids

        self.ids = self.coco.getImgIds(catIds=self.category_ids)
        if annotation_filter:
            self.filter_for_annotations(min_kp_anns=min_kp_anns)
        elif min_kp_anns:
            raise Exception('only set min_kp_anns with annotation_filter')
        LOG.info('Images: %d', len(self.ids))

        self.preprocess = preprocess or openpifpaf.transforms.EVAL_TRANSFORM

    def filter_for_annotations(self, *, min_kp_anns=0):
        LOG.info('filter for annotations (min kp=%d) ...', min_kp_anns)

        def filter_image(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            anns = [ann for ann in anns if not ann.get('iscrowd')]
            if not anns:
                return False
            kp_anns = [ann for ann in anns
                       if 'keypoints' in ann and any(v > 0.0 for v in ann['keypoints'][2::3])]
            return len(kp_anns) >= min_kp_anns

        self.ids = [image_id for image_id in self.ids if filter_image(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)

        category_image_counts = defaultdict(int)
        image_categories = defaultdict(set)
        for ann in anns:
            if ann['iscrowd']:
                continue
            image = ann['image_id']
            category = ann['category_id']
            if category in image_categories[image]:
                continue
            image_categories[image].add(category)
            category_image_counts[category] += 1

        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))

        return weights

    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        local_file_path = os.path.join(self.image_dir, image_info['file_name'])
        with open(local_file_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'local_file_path': local_file_path,
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and annotations
        image, anns, meta = self.preprocess(image, anns, meta)

        LOG.debug(meta)

        # TODO: convert into transform
        # # log stats
        # for ann in anns:
        #     if getattr(ann, 'iscrowd', False):
        #         continue
        #     if not np.any(ann['keypoints'][:, 2] > 0.0):
        #         continue
        #     STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        return image, anns, meta

    def __len__(self):
        return len(self.ids)
