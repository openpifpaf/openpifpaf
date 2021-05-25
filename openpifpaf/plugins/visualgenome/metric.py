from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

from openpifpaf.metric.base import Base
try:
    print("To test scene graph generation, download the evaluator from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch")
    from maskrcnn_benchmark.data.datasets.evaluation.vg.sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall
except:
    pass

import numpy as np
import copy

class VG(Base):
    text_labels_bbox = ['AP', 'AP0.5', 'AP0.75', 'APS', 'APM', 'APL',
                        'ART1', 'ART10', 'AR', 'ARS', 'ARM', 'ARL']

    def __init__(self, obj_categories, rel_categories, mode, iou_types=['bbox', 'relations'], vg_eval=None):


        attribute_on = False #cfg.MODEL.ATTRIBUTE_ON
        num_attributes = 201 #cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_category = len(rel_categories) + 1
        multiple_preds = False #cfg.TEST.RELATION.MULTIPLE_PREDS
        iou_thres = 0.5 #cfg.TEST.RELATION.IOU_THRESHOLD
        self.mode = mode
        self.vg_eval = vg_eval

        assert mode in ['predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet']
        self.iou_types = iou_types
        self.bbox_anns_gt = []
        self.bbox_anns_pred = []
        self.image_ids = []
        #self.rel_anns = []
        predicate_to_ind = {rel:(rel_idx+1) for rel_idx, rel in enumerate(rel_categories)}
        predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

        class_to_ind = {obj:(obj_idx+1) for obj_idx, obj in enumerate(obj_categories)}
        class_to_ind['__background__'] = 0
        self.ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
        self.image_annotations = {}
        if 'relations' in self.iou_types:

            result_dict = {}
            self.evaluator = {}
            # tradictional Recall@K
            eval_recall = SGRecall(result_dict)
            eval_recall.register_container(mode)
            self.evaluator['eval_recall'] = eval_recall

            # no graphical constraint
            eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
            eval_nog_recall.register_container(mode)
            self.evaluator['eval_nog_recall'] = eval_nog_recall

            # test on different distribution
            # eval_zeroshot_recall = SGZeroShotRecall(result_dict)
            # eval_zeroshot_recall.register_container(mode)
            # self.evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall
            #
            # # test on no graph constraint zero-shot recall
            # eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
            # eval_ng_zeroshot_recall.register_container(mode)
            # self.evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall

            # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
            eval_pair_accuracy = SGPairAccuracy(result_dict)
            eval_pair_accuracy.register_container(mode)
            self.evaluator['eval_pair_accuracy'] = eval_pair_accuracy

            # used for meanRecall@K
            eval_mean_recall = SGMeanRecall(result_dict, self.num_rel_category, self.ind_to_predicates, print_detail=True)
            eval_mean_recall.register_container(mode)
            self.evaluator['eval_mean_recall'] = eval_mean_recall

            # used for no graph constraint mean Recall@K
            eval_ng_mean_recall = SGNGMeanRecall(result_dict, self.num_rel_category, self.ind_to_predicates, print_detail=True)
            eval_ng_mean_recall.register_container(mode)
            self.evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

            # prepare all inputs
            self.global_container = {}
            #self.global_container['zeroshot_triplet'] = zeroshot_triplet
            self.global_container['result_dict'] = result_dict
            self.global_container['mode'] = mode
            self.global_container['multiple_preds'] = multiple_preds
            self.global_container['num_rel_category'] = self.num_rel_category
            self.global_container['iou_thres'] = iou_thres
            self.global_container['attribute_on'] = attribute_on
            self.global_container['num_attributes'] = num_attributes

    def verify_anns(self, obj_boxes, obj_labels, ground_truth):
        for pred in ground_truth:
            idx = pred['id']
            box_gt_temp = copy.deepcopy(obj_boxes[idx])
            box_gt_temp[2:] = box_gt_temp[2:] - box_gt_temp[:2]
            label_gt = int(obj_labels[idx])
            assert label_gt == int(pred['category_id'])
            assert np.all(np.equal(box_gt_temp,pred['bbox']))

    def accumulate_bad(self, predictions, image_meta, ground_truth=None):
        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        # self.image_annotations[image_id] = (copy.deepcopy(predictions),
        #                                     copy.deepcopy(image_meta),
        #                                     copy.deepcopy(ground_truth))
        self.image_annotations[image_id] = (predictions,
                                            image_meta,
                                            ground_truth)

    def accumulate(self, predictions, image_meta, ground_truth=None):
        if len(self.iou_types) == 1 and ('bbox' in self.iou_types):
            predictions_det = predictions
        else:
            predictions_rel, predictions_det = predictions

        image_id = int(image_meta['image_id'])
        width, height = image_meta['width_height']
        self.image_ids.append(image_id)

        box = []
        bbox_xyxy = []
        score = []
        label = []

        if ground_truth:
            ground_truth_indices = sorted(range(len(ground_truth)), key=lambda k: ground_truth[k]['id'])
            ground_truth = np.asarray(ground_truth)[ground_truth_indices]
            if self.mode in ['predcls', 'sgcls']:
                predictions_det = np.asarray(predictions_det)[ground_truth_indices]

        for pred in predictions_det:
            box.append(pred.bbox)
            score.append(pred.score)
            label.append(pred.category_id)
            bbox_temp = np.copy(pred.bbox)
            bbox_temp[2:] = bbox_temp[:2] + bbox_temp[2:]
            bbox_xyxy.append(bbox_temp)

        if self.vg_eval:
            gt_idx = self.vg_eval.image_ids.index(image_id)
            obj_boxes = self.vg_eval.gt_boxes[gt_idx].copy()
            obj_labels = self.vg_eval.gt_classes[gt_idx].copy()
            obj_relation_triplets = self.vg_eval.relationships[gt_idx].copy()

        if 'bbox' in self.iou_types:
            if not self.vg_eval:
                for pred in ground_truth:
                    self.bbox_anns_gt.append({
                        'area': pred['area'],
                        'bbox': np.asarray(pred['bbox']), # xywh
                        'category_id': pred['category_id'],
                        'id': len(self.bbox_anns_gt),
                        'image_id': image_id,
                        'iscrowd': 0,
                    })
            else:
                #self.verify_anns(obj_boxes, obj_labels, ground_truth)
                for bbox, category_id in zip(obj_boxes, obj_labels):
                    bbox_temp = copy.deepcopy(bbox)
                    bbox_temp[2:] = bbox_temp[2:] - bbox_temp[:2]
                    self.bbox_anns_gt.append({
                        'area': bbox_temp[2]*bbox_temp[3],
                        'bbox': bbox_temp, # xywh
                        'category_id': category_id,
                        'id': len(self.bbox_anns_gt),
                        'image_id': image_id,
                        'iscrowd': 0,
                    })

            image_id_pred = np.asarray([image_id]*len(box))


            if len(predictions_det)>0:
                self.bbox_anns_pred.append(
                    np.column_stack((image_id_pred, box, score, label))
                )

        if 'relations' in self.iou_types:
            gt_rels = []
            gt_dets_bbox = []
            gt_dets_classes = []
            if not self.vg_eval:
                for s_idx, pred in enumerate(ground_truth):
                    bbox_temp = np.copy(pred['bbox'])
                    bbox_temp[2:] += bbox_temp[:2]
                    gt_dets_bbox.append(bbox_temp)
                    gt_dets_classes.append(pred['category_id'])
                    for rel_idx, rel in enumerate(pred['predicate']):
                        o_idx = pred['object_index'][rel_idx]
                        if ground_truth_indices:
                            o_idx = ground_truth_indices.index(pred['object_index'][rel_idx])
                        gt_rels.append([s_idx, o_idx, rel+1])

            else:
                for rel_idx, rel in enumerate(obj_relation_triplets):
                    gt_rels.append(rel)

                gt_dets_bbox = np.asarray(obj_boxes)
                gt_dets_classes = np.asarray(obj_labels)

            gt_rels = np.asarray(gt_rels)

            rel_anns_idxs = []
            rel_anns_rels = []

            for pred in predictions_rel:
                s_idx = pred.idx_subj
                o_idx = pred.idx_obj
                if ground_truth_indices and self.mode in ['predcls', 'sgcls']:
                    s_idx = ground_truth_indices.index(s_idx)
                    o_idx = ground_truth_indices.index(o_idx)
                rel_anns_idxs.append([int(s_idx), int(o_idx)])
                rel_anns_rels.append(np.insert(pred.rel, 0, 0, axis=0))

            if len(predictions_rel) == 0:
                rel_anns_idxs.append([len(predictions_det)-1, len(predictions_det)-1])
                rel_temp = np.zeros(self.num_rel_category)
                rel_temp[0] = 1
                rel_anns_rels.append(rel_temp)

            self.evaluate_relation_of_one_image((gt_rels, [gt_dets_bbox, gt_dets_classes]), (rel_anns_idxs, rel_anns_rels, bbox_xyxy, score, label), self.global_container, self.evaluator)

    def evaluate_sgg(self):
        for predictions, image_meta, ground_truth in self.image_annotations.values():
            if len(self.iou_types) == 1 and ('bbox' in self.iou_types):
                predictions_det = predictions
            else:
                predictions_rel, predictions_det = predictions
            image_id = int(image_meta['image_id'])
            box = []
            bbox_xyxy = []
            score = []
            label = []

            ground_truth_indices = sorted(range(len(ground_truth)), key=lambda k: ground_truth[k]['id'])

            if ground_truth_indices:
                ground_truth = np.asarray(ground_truth)[ground_truth_indices]
                if self.mode in ['predcls', 'sgcls']:
                    predictions_det = np.asarray(predictions_det)[ground_truth_indices]

            for pred in predictions_det:
                box.append(pred.bbox)
                score.append(pred.score)
                label.append(pred.category_id)
                bbox_temp = np.copy(pred.bbox)
                bbox_temp[2:] = bbox_temp[:2] + bbox_temp[2:]
                bbox_xyxy.append(bbox_temp)

            if self.vg_eval:
                gt_idx = self.vg_eval.image_ids.index(image_id)
                obj_boxes = self.vg_eval.gt_boxes[gt_idx].copy()
                obj_labels = self.vg_eval.gt_classes[gt_idx].copy()
                obj_relation_triplets = self.vg_eval.relationships[gt_idx].copy()

            if 'bbox' in self.iou_types:
                if not self.vg_eval:
                    for pred in ground_truth:
                        self.bbox_anns_gt.append({
                            'area': pred['area'],
                            'bbox': np.asarray(pred['bbox']), # xywh
                            'category_id': pred['category_id'],
                            'id': len(self.bbox_anns_gt),
                            'image_id': image_id,
                            'iscrowd': 0,
                        })
                else:
                    #self.verify_anns(obj_boxes, obj_labels, ground_truth)
                    for bbox, category_id in zip(obj_boxes, obj_labels):
                        bbox_temp = copy.deepcopy(bbox)
                        bbox_temp[2:] = bbox_temp[2:] - bbox_temp[:2]
                        self.bbox_anns_gt.append({
                            'area': bbox_temp[2]*bbox_temp[3],
                            'bbox': bbox_temp, # xywh
                            'category_id': category_id,
                            'id': len(self.bbox_anns_gt),
                            'image_id': image_id,
                            'iscrowd': 0,
                        })

                image_id_pred = np.asarray([image_id]*len(box))


                if len(predictions_det)>0:
                    self.bbox_anns_pred.append(
                        np.column_stack((image_id_pred, box, score, label))
                    )

            if 'relations' in self.iou_types:
                gt_rels = []
                gt_dets_bbox = []
                gt_dets_classes = []
                if not self.vg_eval:
                    for s_idx, pred in enumerate(ground_truth):
                        bbox_temp = np.copy(pred['bbox'])
                        bbox_temp[2:] += bbox_temp[:2]
                        gt_dets_bbox.append(bbox_temp)
                        gt_dets_classes.append(pred['category_id'])
                        for rel_idx, rel in enumerate(pred['predicate']):
                            o_idx = pred['object_index'][rel_idx]
                            if ground_truth_indices:
                                o_idx = ground_truth_indices.index(pred['object_index'][rel_idx])
                            gt_rels.append([s_idx, o_idx, rel+1])

                else:
                    for rel_idx, rel in enumerate(obj_relation_triplets):
                        gt_rels.append(rel)

                    gt_dets_bbox = np.asarray(obj_boxes)
                    gt_dets_classes = np.asarray(obj_labels)

                gt_rels = np.asarray(gt_rels)

                rel_anns_idxs = []
                rel_anns_rels = []

                for pred in predictions_rel:
                    s_idx = pred.idx_subj
                    o_idx = pred.idx_obj
                    if ground_truth_indices and self.mode in ['predcls', 'sgcls']:
                        s_idx = ground_truth_indices.index(s_idx)
                        o_idx = ground_truth_indices.index(o_idx)
                    rel_anns_idxs.append([int(s_idx), int(o_idx)])
                    rel_anns_rels.append(np.insert(pred.rel, 0, 0, axis=0))

                if len(predictions_rel) == 0:
                    rel_anns_idxs.append([len(predictions_det)-1, len(predictions_det)-1])
                    rel_temp = np.zeros(self.num_rel_category)
                    rel_temp[0] = 1
                    rel_anns_rels.append(rel_temp)

                self.evaluate_relation_of_one_image((gt_rels, [gt_dets_bbox, gt_dets_classes]), (rel_anns_idxs, rel_anns_rels, bbox_xyxy, score, label), self.global_container, self.evaluator)

    def _stats_det(self):
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in self.image_ids],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(self.ind_to_classes) if name != '__background__'
                ],
            'annotations': self.bbox_anns_gt,
        }

        fauxcoco.createIndex()
        cocolike_predictions = np.concatenate(self.bbox_anns_pred, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(self.image_ids)#list(range(len(self.image_ids)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def _stats_rel(self):
        # calculate mean recall

        result_str = '\n' + '=' * 100 + '\n'
        self.evaluator['eval_mean_recall'].calculate_mean_recall(self.mode )
        self.evaluator['eval_ng_mean_recall'].calculate_mean_recall(self.mode )

        # print result
        result_str += self.evaluator['eval_recall'].generate_print_string(self.mode )
        result_str += self.evaluator['eval_nog_recall'].generate_print_string(self.mode )
        #result_str += eval_zeroshot_recall.generate_print_string(mode)
        #result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
        result_str += self.evaluator['eval_mean_recall'].generate_print_string(self.mode )
        result_str += self.evaluator['eval_ng_mean_recall'].generate_print_string(self.mode )

        # if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        #     result_str += evaluator['eval_pair_accuracy'].generate_print_string(mode)
        result_str += '=' * 100 + '\n'

        return result_str

    def stats(self):
        data = {}

        #self.evaluate_sgg()
        if 'bbox' in self.iou_types:
            stats_det = self._stats_det()
            data = {
                'stats_det': stats_det.tolist(),
                'text_labels_det': self.text_labels_bbox,
                #'stats_rel': stats_rel,
            }
        if 'relations' in self.iou_types:
            stats_rel = self._stats_rel()
            print(stats_rel)



        return data
    def evaluate_relation_of_one_image(self, groundtruth, prediction, global_container, evaluator):
        """
        Returns:
            pred_to_gt: Matching from predicate to GT
            pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
            pred_triplet_scores: [cls_0score, relscore, cls1_score]
        """
        #unpack all inputs
        mode = global_container['mode']

        local_container = {}
        local_container['gt_rels'] = groundtruth[0]

        # if there is no gt relations for current image, then skip it
        if len(local_container['gt_rels']) == 0:
            return

        local_container['gt_boxes'] = np.asarray(groundtruth[1][0])                  # (#gt_objs, 4)
        local_container['gt_classes'] = np.asarray(groundtruth[1][1])           # (#gt_objs, )

        # about relations
        local_container['pred_rel_inds'] = np.asarray(prediction[0])  # (#pred_rels, 2)
        local_container['rel_scores'] = np.asarray(prediction[1])         # (#pred_rels, num_pred_class)

        # about objects
        local_container['pred_boxes'] = np.asarray(prediction[2])                  # (#pred_objs, 4)
        local_container['pred_classes'] = np.asarray(prediction[4])  # (#pred_objs, )
        local_container['obj_scores'] = np.asarray(prediction[3])            # (#pred_objs, )

        # to calculate accuracy, only consider those gt pairs
        # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing"
        # for sgcls and predcls
        if mode != 'sgdet':
            evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

        # to calculate the prior label based on statistics
        #evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
        #evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

        if mode == 'predcls':
            local_container['pred_boxes'] = local_container['gt_boxes']
            local_container['pred_classes'] = local_container['gt_classes']
            local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

        elif mode == 'sgcls':
            if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
                print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
        elif mode == 'sgdet' or mode == 'phrdet':
            pass
        else:
            raise ValueError('invalid mode')
        """
        elif mode == 'preddet':
            # Only extract the indices that appear in GT
            prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
            if prc.size == 0:
                for k in result_dict[mode + '_recall']:
                    result_dict[mode + '_recall'][k].append(0.0)
                return None, None, None
            pred_inds_per_gt = prc.argmax(0)
            pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
            rel_scores = rel_scores[pred_inds_per_gt]
            # Now sort the matching ones
            rel_scores_sorted = argsort_desc(rel_scores[:,1:])
            rel_scores_sorted[:,1] += 1
            rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))
            matches = intersect_2d(rel_scores_sorted, gt_rels)
            for k in result_dict[mode + '_recall']:
                rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
                result_dict[mode + '_recall'][k].append(rec_i)
            return None, None, None
        """

        if local_container['pred_rel_inds'].shape[0] == 0:
            return

        # Traditional Metric with Graph Constraint
        # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
        local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

        # No Graph Constraint
        evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
        # GT Pair Accuracy
        evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
        # Mean Recall
        evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        # No Graph Constraint Mean Recall
        evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
        # Zero shot Recall
        #evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
        # No Graph Constraint Zero-Shot Recall
        #evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

        return
