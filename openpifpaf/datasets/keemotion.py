from collections import defaultdict
import copy
import logging
from openpifpaf.transforms import annotations
import os

import numpy as np
import torch.utils.data
from PIL import Image

import glob

from .. import transforms, utils

import scipy.ndimage
import imageio

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Keemotion(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    map_categories = {1:1,3:37}

    def __init__(self, image_dir, split='train', *,
                 target_transforms=None,
                 preprocess=None,
                #  n_images=None,
                 config='cifcent',
                 ):

        self.image_dir = image_dir
        self.split = split

        self.images = glob.glob(
            os.path.join(self.image_dir,"images_trainvaltest", self.split, "*.png"),
            recursive=True)
        self.images.sort()

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms
        self.config = config
        self.ball = 'ball' in self.config
        return
        if category_ids is None:
            category_ids = [1]

        self.config = config
        print(self.config)


        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        if ann_inst_file is not None:
            self.coco_inst = COCO(ann_inst_file)

        self.category_ids = category_ids
        print(self.category_ids)
        self.ann_inst_file = ann_inst_file

        if image_filter == 'all':
            self.ids = self.coco.getImgIds()
        elif image_filter == 'annotated':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.filter_for_annotations()
        elif image_filter == 'keypoint-annotations':
            self.ids_kp = self.coco_kp.getImgIds(catIds=self.category_ids)
            self.filter_for_keypoint_annotations()
        elif image_filter == 'kp_inst':
            # print(self.category_ids)

            if self.category_ids == [1]:
                self.ids = self.coco.getImgIds(catIds=self.category_ids)
                # self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids)
                self.filter_for_keypoint_annotations()
            elif self.category_ids == [37]:
                self.ids_ball = self.coco_inst.getImgIds(catIds=self.category_ids)
                self.ids = self.ids_ball
                self.filter_for_ball_annotations()
            else:
                self.ids = self.coco.getImgIds(catIds=self.category_ids[0])
                # self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids[0])
                self.ids_ball = self.coco_inst.getImgIds(catIds=self.category_ids[1])
                self.filter_for_kp_ball_annotations()
                self.ids += self.ids_ball
                # for i in self.ids_ball:
                #     if i not in self.ids:
                #         self.ids.append(i)

                self.ids = list(dict.fromkeys(self.ids))        ## remove duplicate image Ids

            # self.filter_for_annotations()
            # print(self.category_ids)
            # print(len(self.ids))

            # print(len(self.ids_inst))
            # self.filter_for_annotations()
            ### AMA union of kp and inst annotations
            # self.ids_ = []
            # for idx in self.ids:
            #     if idx in self.ids_inst:
            #         self.ids_.append(idx)
            # self.ids = self.ids_


            # if self.category_ids != [1]:

            #     self.ids += self.ids_ball

            #     self.ids = list(dict.fromkeys(self.ids))        ## remove duplicate image Ids

            # self.filter_for_keypoint_annotations_inst()
        else:
            raise Exception('unknown value for image_filter: {}'.format(image_filter))

        # self.ids = [363469]     # for a debug

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))


    def __getitem__(self, index):
        path = self.images[index]
        annotation_path = path.replace('images_trainvaltest/'+self.split, 'panoptic') \
                              .replace('.png', '_panoptic.png')
        image = Image.open(path).convert('RGB')
        annotation = np.array(Image.open(annotation_path)).astype(np.uint32)
        annotation = annotation[:,:,0]+256*(annotation[:,:,1]+256*annotation[:,:,2])

        H, W = annotation.shape
        meshgrid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        meshgrid = np.stack(meshgrid, axis=-1)

        anns = []
        meta = {}

        for instance_id in np.unique(annotation):
            if instance_id < 1000:
                continue
            label = instance_id // 1000
            category_id = self.map_categories[label]

            iid = instance_id % 1000
            mask = annotation == instance_id
            is_crowd = iid == 0

            coords = meshgrid[mask,:]
            center = tuple(coords.mean(axis=0))[::-1]
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            w, h = x2-x1, y2-y1
            x, y = x1+w/2, y1+h/2
            bbox = (x, y, w, h)

            keypoints = np.zeros((18+self.ball,3))
            if label == 1:
                keypoints[17,:] = (*center, 1)
            elif label == 3 and self.ball:
                keypoints[18,:] = (*center, 1)
            else:
                raise NotImplementedError('Class label %d'%label)

            anns.append({
                'num_keypoints': 1,
                'area': coords.shape[0],
                'iscrowd': is_crowd,
                'bmask': mask.astype(np.int64),
                'keypoints': keypoints,
                'image_id': path,
                'id': instance_id,
                'category_id': category_id,
                'bbox_original': bbox,
                'bbox': bbox,
            })


        image, anns, meta = self.preprocess(image, anns, meta)

        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta



        
        image_id = self.ids[index]

        image_info = self.coco.loadImgs(image_id)[0]
        LOG.debug(image_info)
        with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_info['file_name'],
        }

        if 'flickr_url' in image_info:
            _, flickr_file_name = image_info['flickr_url'].rsplit('/', maxsplit=1)
            flickr_id, _ = flickr_file_name.split('_', maxsplit=1)
            meta['flickr_full_page'] = 'http://flickr.com/photo.gne?id={}'.format(flickr_id)

        # anns = []
        # mask = []

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        assert len(anns) != 0, anns


        # mask = []

        if self.ann_inst_file is not None:
            # ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[1])
            # anns_inst = self.coco_inst.loadAnns(ann_ids_inst)

            for i in anns:
                # ann_mask_id = i['id']
                # print(i)
                # print(ann_mask_id)
                # print(np.max(np.max(self.coco_inst.annToMask(i))))
                # print(np.max(np.max(self.coco_inst.annToMask(i) * ann_mask_id)))
                i['bmask'] = self.coco_inst.annToMask(i)
                # mask.append(self.coco_inst.annToMask(i))

        if self.config == 'cif':
            pass

        elif self.config == 'cifcent':
            anns = self.add_center(anns)

        elif self.config == 'cifball':

            anns = self.add_center(anns, visiblity=0)        # add fake ball keypoint

            ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
            anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
            for i in anns_inst:
                # ann_mask_id = i['id']
                i['bmask'] = self.coco_inst.annToMask(i)
                # mask_ball.append(self.coco_inst.annToMask(i) * ann_mask_id)

            if len(anns_inst) is not 0:

                anns_ball = self.empty_person_keypoint(anns_inst)     # add fake people

                anns_ball = self.add_center(anns_ball)        # add ball keypoint

                anns += anns_ball

        elif self.config == 'cifcentball':
            anns = self.add_center(anns)
            anns = self.add_center(anns, visiblity=0)        # add fake ball keypoint
            # mask_ball = []
            ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
            anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
            for i in anns_inst:
                # ann_mask_id = i['id']
                i['bmask'] = self.coco_inst.annToMask(i)
                # mask_ball.append(self.coco_inst.annToMask(i) * ann_mask_id)

            anns_ball = self.empty_person_keypoint(anns_inst, n_keypoints=18)     # add fake people
            anns_ball = self.add_center(anns_ball)        # add ball keypoint
            anns += anns_ball

        else:
            raise NotImplementedError


        # print(len(anns))
        # print(len(mask))
        # anns = self.empty_person_keypoint(mask, image_id)

        anns = copy.deepcopy(anns)

        image, anns, meta = self.preprocess(image, anns, meta)
        # if len(anns) == 0:
        #     import pickle
        #     with open('preprocess.pickle','wb') as f:
        #         pickle.dump(((image, anns, meta),(image_copy, anns_copy, meta_copy)),f)

        # assert len(anns) == len(anns_copy), str(len(anns)) +' '+ str(len(anns_copy))

        # print('222')
        # print(len(anns))
        # print(anns[0].shape)
        # print(image)
        # print(anns)
        # print(meta)
        # mask valid TODO still necessary?
        # print('after augmentation')

        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # print('______valid area______')
        # print(valid_area)
        # print(image.shape)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})

        # transform targets
        if self.target_transforms is not None:
            anns = [t(image, anns, meta) for t in self.target_transforms]

        return image, anns, meta

    def __len__(self):
        return len(self.images)

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotation(image_id)]
        LOG.info('... done.')

    def filter_for_ball_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_ball_annotation(image_id):
            ann_ids = self.coco_inst.getAnnIds(imgIds=image_id)
            anns = self.coco_inst.loadAnns(ann_ids)
            if len(anns) == 0:
                return False
            return True

        self.ids = [image_id for image_id in self.ids if has_ball_annotation(image_id)]
        LOG.info('... done.')

    def filter_for_kp_ball_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False
        def has_ball_annotation(image_id):
            ann_ids = self.coco_inst.getAnnIds(imgIds=image_id)
            anns = self.coco_inst.loadAnns(ann_ids)
            if len(anns) == 0:
                return False
            return True

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotation(image_id) or has_ball_annotation(image_id)]
        LOG.info('... done.')

    ###

    def filter_for_keypoint_annotations_inst(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation_inst(image_id):
            ann_ids = self.coco_inst.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco_inst.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids_inst = [image_id for image_id in self.ids_inst if has_keypoint_annotation_inst(image_id)]
        LOG.info('... done.')

    def filter_for_annotations(self):
        """removes images that only contain crowd annotations"""
        LOG.info('filter for annotations ...')
        def has_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if ann.get('iscrowd'):
                    continue
                return True
            return False

        self.ids = [image_id for image_id in self.ids
                    if has_annotation(image_id)]
        LOG.info('... done.')

    def class_aware_sample_weights(self, max_multiple=10.0):
        """Class aware sampling.

        To be used with PyTorch's WeightedRandomSampler.

        Reference: Solution for Large-Scale Hierarchical Object Detection
        Datasets with Incomplete Annotation and Data Imbalance
        Yuan Gao, Xingyuan Bu, Yang Hu, Hui Shen, Ti Bai, Xubin Li and Shilei Wen
        """
        ann_ids = self.coco.getAnnIds(imgIds=self.ids, catIds = self.category_ids)
        anns = self.coco.loadAnns(ann_ids)
        # print(len(anns))

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
        # print(category_image_counts)
        weights = [
            sum(
                1.0 / category_image_counts[category_id]
                for category_id in image_categories[image_id]
            )
            for image_id in self.ids
        ]
        # print(weights)
        min_w = min(weights)
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        max_w = min_w * max_multiple
        weights = [min(w, max_w) for w in weights]
        LOG.debug('Class Aware Sampling: minW = %f, maxW = %f', min_w, max(weights))
        # print(min_w)
        # print(max_w)
        return weights

    # def add_center_person(self, anns_center, visiblity=2):

    #     anns_copy = copy.deepcopy(anns_center)
    #     counter = 0
    #     for ann in anns_center:
    #         meshgrid = np.indices(ann['bmask'].shape)
    #         meshgrid[0] *= ann['bmask']
    #         meshgrid[1] *= ann['bmask']
    #         center = (meshgrid[0].sum()/ann['bmask'].sum(),
    #                 meshgrid[1].sum()/ann['bmask'].sum())


    #         counter += 1
    #         ann['keypoints'].append(int(center[1]))      # add center for y
    #         counter += 1
    #         ann['keypoints'].append(int(center[0]))      # add center for x
    #         counter += 1
    #         ann['keypoints'].append(visiblity)

    #     assert counter == len(anns_center) * 3, counter
    #     for ann in anns_center:
    #         if len(ann['keypoints']) != 54:
    #             import pickle
    #             if not os.path.isfile('add_center.pickle'):
    #                 with open('add_center.pickle','wb') as f:
    #                     pickle.dump((anns_center, anns_copy),f)
    #         assert len(ann['keypoints']) == 54, len(ann['keypoints'])

    #     return anns_center

    def add_center(self, anns_center, visiblity=2):

        assert visiblity == 2 or visiblity == 0, visiblity
        # for ann in anns_center:
        #     if len(ann['keypoints']) != 51:
        #         import pickle
        #         if not os.path.isfile('add_center.pickle'):
        #             with open('add_center.pickle','wb') as f:
        #                 pickle.dump(anns_center,f)
        # counter = 0

        # anns_copy = copy.deepcopy(anns_center)
        for ann_id, ann in enumerate(anns_center):
            meshgrid = np.indices(anns_center[ann_id]['bmask'].shape)
            meshgrid[0] *= anns_center[ann_id]['bmask']
            meshgrid[1] *= anns_center[ann_id]['bmask']
            center = (meshgrid[0].sum()/anns_center[ann_id]['bmask'].sum(),
                    meshgrid[1].sum()/anns_center[ann_id]['bmask'].sum())

            # assert isinstance(center[1], float)
            # assert isinstance(center[0], float)

            # assert len(ann['keypoints']) == 51, len(ann['keypoints'])

            # keypoints = copy.deepcopy(ann['keypoints'])
            # counter += 1
            anns_center[ann_id]['keypoints'].append(int(center[1]))      # add center for y
            # counter += 1
            anns_center[ann_id]['keypoints'].append(int(center[0]))      # add center for x
            # counter += 1
            anns_center[ann_id]['keypoints'].append(visiblity)

        # assert counter == len(anns_center) * 3, counter
        # for ann in anns_center:
        #     if len(ann['keypoints']) != 54:
        #         import pickle
        #         # if not os.path.isfile('add_center.pickle'):
        #         with open('add_center.pickle','wb') as f:
        #             pickle.dump((anns_center, anns_copy, image, image_id),f)
            # assert len(ann['keypoints']) == 54, len(ann['keypoints'])



        return anns_center

    def empty_person_keypoint(self, anns_inst, n_keypoints=17, category_id=37):


        for ann in anns_inst:
            keypoints = []
            for _ in range(3 * n_keypoints):
                keypoints.append(0)
            ann['keypoints'] = keypoints

        return anns_inst

