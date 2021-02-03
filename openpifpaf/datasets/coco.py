from collections import defaultdict
import copy
import logging
import os

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils

import scipy.ndimage

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class Coco(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, image_dir, ann_file, *, ann_inst_file=None, target_transforms=None,
                 n_images=None, preprocess=None,
                 category_ids=None,
                 image_filter='keypoint-annotations'):
        if category_ids is None:
            category_ids = [1]

        from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        if ann_inst_file is not None:
            self.coco_inst = COCO(ann_inst_file)

        self.category_ids = category_ids
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
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids)
            print(len(self.ids))
            print(len(self.ids_inst))
            # self.filter_for_annotations()
            ### AMA union of kp and inst annotations
            self.ids_ = []
            for idx in self.ids:
                if idx in self.ids_inst:
                    self.ids_.append(idx)
            self.ids = self.ids_
            self.filter_for_keypoint_annotations()
            self.filter_for_keypoint_annotations_inst()
        else:
            raise Exception('unknown value for image_filter: {}'.format(image_filter))

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))

        

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotation(image_id)]
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
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
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

    def add_center(self, image, anns, mask):
        # print(len(anns))
        # print(len(mask))
        
        for id_m, msk in enumerate(mask):
            meshgrid = np.indices(msk.shape)
            meshgrid[0] *= msk
            meshgrid[1] *= msk
            center = (meshgrid[0].sum()/msk.sum(),
                    meshgrid[1].sum()/msk.sum())

            # print(len(anns))
            keypoints = anns[id_m]['keypoints']
            # print('start of process')
            # print(sum(keypoints))
            # if sum(keypoints) == 0:
            #     import pickle
            #     import os.path
            #     from os import path
            #     if path.isfile('keypoints_emp.pickle') is not True:
            #         print('file saved')
            #         with open('keypoints_emp.pickle','wb') as f:
            #             pickle.dump((image,anns,mask), f)
                # torch.save((image,anns,mask),'keypoints_emp_coco.pt')
            # print(sum(sum(msk)))
            anns[id_m]['keypoints'].append(int(center[1]))      # add center for x
            anns[id_m]['keypoints'].append(int(center[0]))      # add center for y
            anns[id_m]['keypoints'].append(2)

            # print(int(center[1]))
            # print(int(center[0]))
            # print(2)

            anns[id_m]['keypoints'].append(int(center[1]))      # add center for x
            anns[id_m]['keypoints'].append(int(center[0]))      # add center for y
            anns[id_m]['keypoints'].append(2)
            # keypoints = anns[id_m]['keypoints']
            # if anns[id_m]['keypoints']
            # print(type(keypoints))
            # print(keypoints)
            

        return anns


    def __getitem__(self, index):
        image_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids)
        anns = self.coco.loadAnns(ann_ids)


        mask = []
        if self.ann_inst_file is not None:
            ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=self.category_ids)
            anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
            
            for i in anns_inst:
                mask.append(self.coco_inst.annToMask(i))
                
            assert len(anns) == len(mask)
        
        

        # print('mask')
        # print(mask[0].shape)
        # anns.append(mask)
        # anns[0]['mask'] = mask
        # anns.append(dict(mask = mask))
        anns = copy.deepcopy(anns)
        mask = copy.deepcopy(mask)
        # anns_inst = copy.deepcopy(anns_inst)

        # anns.append(np.squeeze(anns_inst)[0])
        # return anns
        

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
        

        
        # hei = mask[0].shape[0]
        # wid = mask[0].shape[1]
        # mask_copy = mask.copy()


        # ## encode the masks
        # masks = np.zeros((hei, wid))    
        # for i in range(len(mask)):    
        #     masks [mask[i] == 1] = i+1

        # if np.unique(masks).tolist() == [0.0, 1.0]:
        #     torch.save((image,masks,meta),'image_empt_coco.pt')

        # print('coco')

        
        # preprocess image and annotations
        # image, anns, meta = self.preprocess(image, anns, meta)
        
        
        ### AMA
        # print('111')
        # print(image)
        # print(len(anns))
        # print(anns)
        # print(anns[0].shape)
        # print(mask) # a list of masks of people in image
        # print(meta)

        if self.ann_inst_file is not None:
            anns = self.add_center(image,anns, mask)
        
        image, anns, mask, meta = self.preprocess(image, anns, mask, meta)
        # print('222')
        # print(len(anns))
        # print(anns[0].shape)
        # print(image)
        # print(anns)
        # print(meta)
        # mask valid TODO still necessary?
        
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
            anns = [t(image, anns, mask, meta) for t in self.target_transforms]

        
        # hei_wid = mask[0].shape[0]
        # masks = np.zeros((49, 49))
        # # print(mask[0].sum())
        # for i in range(len(mask)):
        #     mask[i] = scipy.ndimage.zoom(mask[i], 49/hei_wid, order=1)
        #     masks [mask[i] == 1] = i+1
        ###
        # # anns.append(mask)
        # print("annnsnsnsnsn")
        # # print(anns.shape)
        # print(len(anns))
        # print(len(anns[0]))
        # # print(anns[0].shape)
        # print(len(anns[0][0]))
        # # print(len(anns[0][0][0]))
        # # print(len(anns[0][0][16]))
        # # print(len(anns[0][1]))
        # # print(len(anns[0][1][0]))
        # # print(len(anns[0][1][16]))        
        # # print(len(anns[0][2]))
        # # print(len(anns[0][2][0]))
        # # print(len(anns[0][2][16]))
        # print(len(anns[1]))
        # # print(anns[1].shape)
        # print(len(anns[1][0]))
        return image, anns, meta

    def __len__(self):
        return len(self.ids)
