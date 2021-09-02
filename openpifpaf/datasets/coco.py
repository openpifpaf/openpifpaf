from collections import defaultdict
import copy
import logging
import os

import numpy as np
import torch.utils.data
from PIL import Image

from .. import transforms, utils

import scipy.ndimage
import matplotlib.pyplot as plt

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
                 image_filter='keypoint-annotations',
                 config='cif',
                 ball=False,
                 filter_for_medium=False,
                 eval_coco=False):
        if category_ids is None:
            category_ids = [1]

        self.config = config
        print(self.config)
        self.ids_ball = []
        self.ball = ball
        self.eval_coco = eval_coco

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
            self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids)
        elif image_filter == 'annotated':
            self.ids = self.coco.getImgIds(catIds=self.category_ids)
            self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids)
            self.filter_for_annotations()
        # elif image_filter == 'keypoint-annotations':
        #     self.ids_kp = self.coco_kp.getImgIds(catIds=self.category_ids)
        #     self.filter_for_keypoint_annotations()
        
        elif image_filter == 'kp_inst':
            if self.category_ids == [1]:
                self.ids = self.coco.getImgIds(catIds=self.category_ids)
                self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids)
                self.filter_for_keypoint_annotations()
                # self.ids += self.ids

            elif self.category_ids == [37]:
                self.ids_ball = self.coco_inst.getImgIds(catIds=self.category_ids)
                self.ids = self.ids_ball
                self.filter_for_ball_annotations()
            else:
                self.ids = self.coco.getImgIds(catIds=self.category_ids[0])
                # self.ids_inst = self.coco_inst.getImgIds(catIds=self.category_ids[0])
                self.ids_ball = self.coco_inst.getImgIds(catIds=self.category_ids[1])
                self.filter_for_keypoint_annotations()
                if filter_for_medium:
                    self.filter_for_medium_people()
                # self.filter_for_kp_ball_annotations()
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

        # self.ids = [327701]     # for a debug
        

        if n_images:
            self.ids = self.ids[:n_images]
        LOG.info('Images: %d', len(self.ids))
        print('Number of images: ', len(self.ids))

        # self.Get_number_of_images_with_ball()
        # if eval_coco:
        #     self.ids = [458755]

        self.preprocess = preprocess or transforms.EVAL_TRANSFORM
        self.target_transforms = target_transforms

    def Get_number_of_images_with_ball(self):
        LOG.info('Images with ball ...')
        def has_ball(image_id):
            ann_ids = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
            anns = self.coco_inst.loadAnns(ann_ids)
            if len(anns) > 0:
                return True
            return False
        self.ids = [image_id for image_id in self.ids
                    if has_ball(image_id)]
        print('Number of images with ball', len([image_id for image_id in self.ids
                    if has_ball(image_id)]))
        LOG.info('... done.')

    def filter_for_medium_people(self):
        LOG.info('filter for medium people ...')
        def is_medium(image_id,n_t_plot):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[1])
            anns = self.coco.loadAnns(ann_ids)
            for ix, ann in enumerate(anns):
                # mask = self.coco_inst.annToMask(ann)
                if 'keypoints' not in ann:
                    continue
                if ann.get('iscrowd'):
                    continue
                mask = self.coco_inst.annToMask(ann)
                if mask.sum() < 0.01 * (np.ones_like(mask)).sum():
                    if ix == len(anns) - 1 and len(anns) > 1:
                        # if n_t_plot:
                        #     print('image id (kept 1):', image_id)
                        #     image_info = self.coco.loadImgs(image_id)[0]
                        #     with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
                        #         image = Image.open(f).convert('RGB')
                        #         plt.imshow(image)
                        #         plt.show()
                        return True
                    else:
                        continue
                if len([v for v in ann['keypoints'][2::3] if v > 0.0]) > 5:
                    if ix == len(anns) - 1:
                        # if n_t_plot:
                        #     print('image id (kept 2):', image_id)
                        #     image_info = self.coco.loadImgs(image_id)[0]
                        #     with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
                        #         image = Image.open(f).convert('RGB')
                        #         plt.imshow(image)
                        #         plt.show()
                        return True
                    else:
                        continue
                # if n_t_plot:
                #     print('image id (1):', image_id)
                #     image_info = self.coco.loadImgs(image_id)[0]
                #     with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
                #         image = Image.open(f).convert('RGB')
                #         plt.imshow(image)
                #         plt.show()
                return False
            # if n_t_plot:
            #     print('image id (2):', image_id)
            #     image_info = self.coco.loadImgs(image_id)[0]
            #     with open(os.path.join(self.image_dir, image_info['file_name']), 'rb') as f:
            #         image = Image.open(f).convert('RGB')
            #         plt.imshow(image)
            #         plt.show()
            return False
        num_to_plot = [False for _ in self.ids]
        for i in range(30):
            num_to_plot[i] = True
        print('number of images before removing large people: ', len(self.ids))
        self.ids = [image_id for image_id, n_t_plot in zip(self.ids, num_to_plot)
                    if is_medium(image_id,n_t_plot)]
        print('number of images after removing large people: ', len(self.ids))
        LOG.info('... done.')

    def filter_for_keypoint_annotations(self):
        LOG.info('filter for keypoint annotations ...')
        self.counter = 0
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=[1])
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    # self.counter += 1
                    # print(self.counter)
                    return True
            return False
        self.ids = [image_id for image_id in self.ids
                    if has_keypoint_annotation(image_id)]
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

    def add_center(self, anns_center, image=None, visiblity=2):
        
        for ann_id, ann in enumerate(anns_center):
            meshgrid = np.indices(anns_center[ann_id]['bmask'].shape)
            meshgrid[0] *= anns_center[ann_id]['bmask']
            meshgrid[1] *= anns_center[ann_id]['bmask']
            center = (meshgrid[0].sum()/anns_center[ann_id]['bmask'].sum(),
                    meshgrid[1].sum()/anns_center[ann_id]['bmask'].sum())

            # anns_center[ann_id]['keypoints'] = 3*17*[0]
            
            anns_center[ann_id]['keypoints'].append(int(center[1]))      # add center for y
            anns_center[ann_id]['keypoints'].append(int(center[0]))      # add center for x
            anns_center[ann_id]['keypoints'].append(visiblity)    

        
        return anns_center

    def add_center_sp(self, anns_center, visiblity=2):
        
        for ann_id, ann in enumerate(anns_center):
            meshgrid = np.indices(anns_center[ann_id]['bmask'].shape)
            meshgrid[0] *= anns_center[ann_id]['bmask']
            meshgrid[1] *= anns_center[ann_id]['bmask']
            center = (meshgrid[0].sum()/anns_center[ann_id]['bmask'].sum(),
                    meshgrid[1].sum()/anns_center[ann_id]['bmask'].sum())
            
            anns_center[ann_id]['cent'] = []
            anns_center[ann_id]['cent'].append(int(center[1]))      # add center for y
            anns_center[ann_id]['cent'].append(int(center[0]))      # add center for x
            anns_center[ann_id]['cent'].append(visiblity)
        
        return anns_center

    def add_ball(self, anns_center, visiblity=2, n_keypoints=18):
        
        for ann_id, ann in enumerate(anns_center):
            meshgrid = np.indices(anns_center[ann_id]['bmask'].shape)
            meshgrid[0] *= anns_center[ann_id]['bmask']
            meshgrid[1] *= anns_center[ann_id]['bmask']
            center = (meshgrid[0].sum()/anns_center[ann_id]['bmask'].sum(),
                    meshgrid[1].sum()/anns_center[ann_id]['bmask'].sum())
            
            anns_center[ann_id]['kp_ball'] = []
            anns_center[ann_id]['kp_ball'].append(int(center[1]))      # add center for y
            anns_center[ann_id]['kp_ball'].append(int(center[0]))      # add center for x
            anns_center[ann_id]['kp_ball'].append(visiblity)
            anns_center[ann_id]['keypoints'] = 3*n_keypoints*[0]
            anns_center[ann_id]['cent'] = 3*[0]
        
        return anns_center

    def empty_person_keypoint(self, anns_inst, n_keypoints=17):
        
        for ann in anns_inst:
            keypoints = []
            for _ in range(3 * n_keypoints):
                keypoints.append(0)
            ann['keypoints'] = keypoints

        return anns_inst
    


    def __getitem__(self, index):

        image_id = self.ids[index]
        # print('------------------------------------------')
        # print('----------- IMAGAE ID: ', image_id)

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

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        anns = copy.deepcopy(anns)
        
        
        # assert anns != []

        if self.ann_inst_file is not None:
            # ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[1])
            # anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
            
            # print id
            
            for i in anns:
                i['bmask'] = self.coco_inst.annToMask(i)

        try:

            if self.config == 'cif':
                pass

            elif self.config == 'cifcent':
                # print('cifcent added')
                anns = self.add_center(anns)

            elif self.config == 'cif cent':
                # print('center added')
                anns = self.add_center_sp(anns)

            elif self.config == 'cifball':

                anns = self.add_center(anns, image=image, visiblity=0)        # add fake ball keypoint

                ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
                anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
                anns_inst = copy.deepcopy(anns_inst)
                for i in anns_inst:
                    # ann_mask_id = i['id']
                    i['bmask'] = self.coco_inst.annToMask(i)
                    # mask_ball.append(self.coco_inst.annToMask(i) * ann_mask_id) 

                if len(anns_inst) is not 0:

                    anns_ball = self.empty_person_keypoint(anns_inst)     # add fake people

                    anns_ball = self.add_center(anns_ball, image=image)        # add ball keypoint

                    anns += anns_ball
                
            elif self.config == 'cifcentball':
                anns = self.add_center(anns)
                anns = self.add_center(anns, visiblity=0)        # add fake ball keypoint
                # mask_ball = []
                ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
                anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
                anns_inst = copy.deepcopy(anns_inst)
                for i in anns_inst:
                    # ann_mask_id = i['id']
                    i['bmask'] = self.coco_inst.annToMask(i)
                    # mask_ball.append(self.coco_inst.annToMask(i) * ann_mask_id)

                anns_ball = self.empty_person_keypoint(anns_inst, n_keypoints=18)     # add fake people
                anns_ball = self.add_center(anns_ball)        # add ball keypoint
                anns += anns_ball

            if self.ball:
                # print('ball added')
                ann_ids_inst = self.coco_inst.getAnnIds(imgIds=image_id, catIds=[37])
                anns_inst = self.coco_inst.loadAnns(ann_ids_inst)
                anns_inst = copy.deepcopy(anns_inst)
                for i in anns_inst:
                    # ann_mask_id = i['id']
                    i['bmask'] = self.coco_inst.annToMask(i)

                for ann in anns:
                    ann['kp_ball'] = [0,0,0]
                    # ann['kp_ball'] = np.zeros((1, 3))
                n_keypoints = 18 if self.config == 'cifcent' else 17
                anns_ball = self.add_ball(anns_inst, n_keypoints=n_keypoints)
                # anns_ball = self.empty_person_keypoint(anns_ball, n_keypoints=18)   # add fake people
                anns += anns_ball
                
        except:
            print('image_id_1: ', image_id)
            import pickle
            with open('coco_1.pickle','wb') as f:
                pickle.dump((image, anns),f)
        # anns = []

        # anns = copy.deepcopy(anns)
        
        # try:
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # plt.figure(figsize=(10,10))
        # print('number of anns',len(anns))
        for aaa in anns:
            n_keypoints = 18 if self.config == 'cifcent' else 17
            assert len(aaa['keypoints']) == 3*n_keypoints, len(aaa['keypoints'])
            # print('---------------- anns', aaa['cent'])
            if self.config == 'cif cent':
                assert len(aaa['cent']) == 3, len(aaa['cent'])
        #     plt.scatter(aaa['kp_ball'][0],aaa['kp_ball'][1],linewidths=4)
            # plt.show
        image, anns, meta = self.preprocess(image, anns, meta)
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)
        # plt.figure(figsize=(10,10))
        # for aaa in anns:
        # print('number of anns',len(anns))
        #     plt.scatter(aaa['kp_ball'][0,0],aaa['kp_ball'][0,1],linewidths=4)
        #     plt.show
        # except:
        #     print('image_id_2: ', image_id)
        #     import pickle
        #     with open('coco_2.pickle','wb') as f:
        #         pickle.dump((image, anns),f)

        # mask valid TODO still necessary?
        # print('after augmentation')
        meta['keypoints_GT'] = []
        for i in anns:
            meta['keypoints_GT'].append(i['keypoints'])
        
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        LOG.debug(meta)

        # log stats
        for ann in anns:
            if getattr(ann, 'iscrowd', False):
                continue
            if not np.any(ann['keypoints'][:, 2] > 0.0):
                continue
            STAT_LOG.debug({'bbox': [int(v) for v in ann['bbox']]})


        # transform targets
        # try:
        #     if self.target_transforms is not None:
        #         anns = [t(image, anns, meta) for t in self.target_transforms]
        # except:
        #     print('image_id_3: ', image_id)
        #     import pickle
        #     with open('coco_3.pickle','wb') as f:
        #         pickle.dump((image, anns),f)
        
        anns_trans = []
        if self.target_transforms is not None:
            for t in self.target_transforms:
                # try:
                anns_trans.append(t(image, anns, meta))
                # except:
                #     print('image_id_3: ', image_id, t)
        if self.eval_coco:
            # print('in coco')
            # print(len(anns_trans[0]))
            # print(anns_trans[0][0].shape)
            # print(anns_trans[0][1].shape)
            # print(anns_trans[0][2].shape)
            # print(len(anns_trans[2]))
            # print(anns_trans[2][0].shape)
            # print(anns_trans[2][1].shape)
            # print(anns_trans[2][2].shape)
            # print(len(anns_trans[3]))
            # print(anns_trans[3][0].shape)
            # print(anns_trans[3][1].shape)
            # print(anns_trans[3][2].shape)
            return image, anns, meta, anns_trans

        anns = anns_trans

        return image, anns, meta

    def __len__(self):
        return len(self.ids)
        # return 50
        