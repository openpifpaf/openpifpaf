import logging
import os
import torch.utils.data
from PIL import Image

from . import transforms


class CocoKeypoints(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Based on `torchvision.dataset.CocoDetection`.

    Caches preprocessing.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, return_meta=False):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        if not all_images:
            self.ids = self.coco.getImgIds(catIds=self.cat_ids)
            self.filter_for_keypoint_annotations()
        else:
            self.ids = self.coco.getImgIds()
        if n_images:
            self.ids = self.ids[:n_images]
        print('Images: {}'.format(len(self.ids)))

        self.preprocess = preprocess or transforms.Preprocess()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms
        self.return_meta = return_meta

        self.log = logging.getLogger(self.__class__.__name__)

    def filter_for_keypoint_annotations(self):
        print('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
        print('... done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        target = self.coco.loadAnns(ann_ids)

        image_info = self.coco.loadImgs(img_id)[0]
        self.log.debug(image_info)
        with open(os.path.join(self.root, image_info['file_name']), 'rb') as f:
            img = Image.open(f).convert('RGB')

        meta = {
            'image_id': img_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # preprocess image and target
        img, target, preprocess_meta = self.preprocess(img, target)
        meta.update(preprocess_meta)

        # transform image
        original_size = img.size
        img = self.image_transform(img)

        # transform targets
        if self.target_transforms is not None:
            target = [t(target, original_size) for t in self.target_transforms]

        if self.return_meta:
            return img, target, meta
        return img, target

    def __len__(self):
        return len(self.ids)
