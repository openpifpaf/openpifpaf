import numpy as np

from ..annotation import Annotation, AnnotationCrowd, AnnotationDet
from .preprocess import Preprocess


class ToAnnotations(Preprocess):
    def __init__(self, converters):
        self.converters = converters

    def __call__(self, image, anns, meta):
        anns = [
            ann
            for converter in self.converters
            for ann in converter(anns)
        ]
        return image, anns, meta


class ToKpAnnotations:
    def __init__(self, categories, keypoints_by_category, skeleton_by_category):
        self.keypoints_by_category = keypoints_by_category
        self.skeleton_by_category = skeleton_by_category
        self.categories = categories

    def __call__(self, anns):
        return [
            Annotation(
                self.keypoints_by_category[ann['category_id']],
                self.skeleton_by_category[ann['category_id']],
                categories=self.categories,
            )
            .set(
                ann['keypoints'],
                category_id=ann['category_id'],
                fixed_score='',
                fixed_bbox=ann.get('bbox'),
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['keypoints'][2::3] > 0.0)
        ]


class ToDetAnnotations:
    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationDet(categories=self.categories)
            .set(
                ann['category_id'],
                None,
                ann['bbox'],
            )
            for ann in anns
            if not ann['iscrowd'] and np.any(ann['bbox'])
        ]


class ToCrowdAnnotations:
    def __init__(self, categories):
        self.categories = categories

    def __call__(self, anns):
        return [
            AnnotationCrowd(categories=self.categories)
            .set(
                ann.get('category_id', 1),
                ann['bbox'],
            )
            for ann in anns
            if ann['iscrowd']
        ]
