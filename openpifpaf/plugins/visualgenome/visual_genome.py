from collections import defaultdict
import copy
import logging
import os
import numpy as np
import h5py
import json
import torch

import torch.utils.data
from PIL import Image

from openpifpaf import transforms, utils


LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

class VG(torch.utils.data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        image_dir (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
    """

    def __init__(self, data_dir,*,
                 preprocess=None, split="train", num_im=-1, num_val_im=-1,
                 filter_duplicate_rels=True, filter_non_overlap=True, filter_empty_rels=True, use_512=False, eval_mode=False):
        assert split == "train" or split == "test", "split must be one of [train, val, test]"
        assert num_im >= -1, "the number of samples must be >= 0"
        self.eval_mode = eval_mode
        # split = 'train' if split == 'test' else 'test'
        self.data_dir = data_dir
        self.preprocess = preprocess

        self.split = split
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'

        self.roidb_file = os.path.join(self.data_dir, "VG-SGG.h5")
        self.use_512 = use_512
        if self.use_512:
            self.image_file = os.path.join(self.data_dir, "imdb_512.h5")
        else:
            self.image_file = os.path.join(self.data_dir, "imdb_1024.h5")
        # read in dataset from a h5 file and a dict (json) file
        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)
        self.im_h5 = h5py.File(self.image_file, 'r')
        self.info = json.load(open(os.path.join(self.data_dir, "VG-SGG-dicts.json"), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        # cfg.ind_to_class = self.ind_to_classes

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
        # cfg.ind_to_predicate = self.ind_to_predicates

        self.split_mask, self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships, self.image_ids = load_graphs(
            self.roidb_file, self.image_file,
            self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=filter_non_overlap and split == "train",
            use_512=self.use_512
        )

        self.json_category_id_to_contiguous_id = self.class_to_ind

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        LOG.info('Images: %d', len(self.image_index))


    def get_frequency_prior(self, obj_categories, rel_categories):
        fg_matrix = np.zeros((
            len(obj_categories),  # not include background
            len(obj_categories),  # not include background
            len(rel_categories),  # include background
        ), dtype=np.int64)

        bg_matrix = np.zeros((
            len(obj_categories),  # not include background
            len(obj_categories),  # not include background
        ), dtype=np.int64)

        smoothing_pred = np.zeros(len(rel_categories), dtype=np.float32)

        count_pred = 0.0
        for index in range(len(self.image_index)):
            # get all object boxes
            gt_box_to_label = {}
            obj_boxes = self.gt_boxes[index].copy()
            obj_labels = self.gt_classes[index].copy()
            obj_relation_triplets = self.relationships[index].copy()
            for i, target in enumerate(obj_relation_triplets):
                subj_id = target[0]
                obj_id = target[1]
                prd_lbl = target[2]-1
                x, y, x2, y2 = obj_boxes[subj_id]
                w, h = x2-x, y2-y
                sbj_lbl = int(obj_labels[subj_id])-1

                sbj_box = [x,y,w,h]
                x1, y1, x2, y2 = obj_boxes[obj_id]
                w1, h1 = x2-x1, y2-y1
                obj_lbl = int(obj_labels[obj_id])-1
                obj_box = [x1,y1,w1,h1]

                if tuple(sbj_box) not in gt_box_to_label:
                    gt_box_to_label[tuple(sbj_box)] = sbj_lbl
                if tuple(obj_box) not in gt_box_to_label:
                    gt_box_to_label[tuple(obj_box)] = obj_lbl


                fg_matrix[sbj_lbl, obj_lbl, prd_lbl] += 1

                for b1, l1 in gt_box_to_label.items():
                    for b2, l2 in gt_box_to_label.items():
                        if b1 == b2:
                            continue
                        bg_matrix[l1, l2] += 1

                smoothing_pred[prd_lbl] += 1.0
                count_pred += 1.0

        return fg_matrix, bg_matrix, (smoothing_pred/count_pred)

    def _im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    def get_img_info(self, img_id):
        w, h = self.im_sizes[img_id, :]
        return {"height": h, "width": w}

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_classes[class_id]

    def __getitem__(self, index):

        # get image
        image = Image.fromarray(self._im_getter(index)); width, height = image.size
        image_id = self.image_ids[index]
        # get object bounding boxes, labels and relations
        obj_boxes = self.gt_boxes[index].copy()
        obj_labels = self.gt_classes[index].copy()
        obj_relation_triplets = self.relationships[index].copy()

        # if self.filter_duplicate_rels:
        #     # Filter out dupes!
        #     assert self.split == 'train'
        #     old_size = obj_relation_triplets.shape[0]
        #     all_rel_sets = defaultdict(list)
        #     for (o0, o1, r) in obj_relation_triplets:
        #         all_rel_sets[(o0, o1)].append(r)
        #     obj_relation_triplets = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
        #     obj_relation_triplets = np.array(obj_relation_triplets)

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = obj_relation_triplets.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in obj_relation_triplets:
                all_rel_sets[(o0, o1, r)].append(1)
            obj_relation_triplets = [(k[0], k[1], k[2]) for k,v in all_rel_sets.items()]
            obj_relation_triplets = np.array(obj_relation_triplets)


        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        local_file_path = os.path.join(self.data_dir, "VG_100K", str(image_id)+".jpg")
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_id,
            'local_file_path': local_file_path,
        }

        anns = []
        dict_counter = {}
        for target in obj_relation_triplets:
            subj_id = target[0]
            obj_id = target[1]
            pred = target[2]
            x, y, x2, y2 = obj_boxes[subj_id]
            w, h = x2-x, y2-y

            if subj_id not in dict_counter:
                dict_counter[subj_id] = len(anns)
                anns.append({
                    'id': subj_id,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(obj_labels[subj_id]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [len(anns) + 1] if obj_id not in dict_counter else [int(dict_counter[obj_id])],
                    'predicate': [pred-1],
                })
            else:
                if obj_id in dict_counter:
                    anns[dict_counter[subj_id]]['object_index'].append(dict_counter[obj_id])
                else:
                    anns[dict_counter[subj_id]]['object_index'].append(len(anns))
                anns[dict_counter[subj_id]]['predicate'].append(pred-1)

            x, y, x2, y2 = obj_boxes[obj_id]
            w, h = x2-x, y2-y

            if obj_id not in dict_counter:
                dict_counter[obj_id] = len(anns)
                anns.append({
                    'id': obj_id,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(obj_labels[obj_id]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [],
                    'predicate': [],
                })
        for idx, det in enumerate(zip(obj_boxes, obj_labels)):
            if idx in dict_counter:
                continue
            x, y, x2, y2 = det[0]
            w, h = x2-x, y2-y
            dict_counter[idx] = len(anns)
            anns.append({
                    'id': idx,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(det[1]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [],
                    'predicate': [],
                    })

        assert len(anns) == len(obj_boxes)
        if self.eval_mode:
            anns_gt = copy.deepcopy(anns)
            image, anns, meta = self.preprocess(image, anns, meta)
        else:
            # preprocess image and annotations
            image, anns, meta = self.preprocess(image, anns, meta)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)
        if self.eval_mode:
            return image, (anns, anns_gt), meta
        return image, anns, meta

    def __len__(self):
        return len(self.image_index)

def load_graphs(graphs_file, images_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False, use_512=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    im_h5 = h5py.File(images_file, 'r')

    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    if use_512:
        all_boxes = roi_h5['boxes_{}'.format(512)][:]  # will index later
    else:
        all_boxes = roi_h5['boxes_{}'.format(1024)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    im_widths = im_h5["image_widths"][split_mask]
    im_heights = im_h5["image_heights"][split_mask]
    im_ids = im_h5["image_ids"][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    im_sizes = []
    image_index_valid = []
    boxes = []
    gt_classes = []
    relationships = []
    image_ids = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(torch.from_numpy(boxes_i).float(), torch.from_numpy(boxes_i).float()).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        image_index_valid.append(image_index[i])
        im_sizes.append(np.array([im_widths[i], im_heights[i]]))
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        image_ids.append(im_ids[i])

    im_sizes = np.stack(im_sizes, 0)
    return split_mask, image_index_valid, im_sizes, boxes, gt_classes, relationships, image_ids
