#ifndef CIF_CAF_HPP
#define CIF_CAF_HPP

#include <iostream>

#include "field_config.hpp"
#include "annotation.hpp"
#include "occupancy.hpp"
#include "caf_scored.hpp"
#include "nms.hpp"

#include <queue>
#include <map>
#include <set>

static vector<Annotation> INITIAL_ANNOTATIONS;


class CifCaf {
    public:
        unsigned H;
        unsigned W;
        Fields fields;
        bool nms;

        string connection_method = "blend";
        bool force_complete = false;
        float force_complete_caf_th = 0.001;
        bool greedy = false;
        float keypoint_threshold = 0.15;
        float keypoint_threshold_rel = 0.5;
        bool nms_before_force_complete = false;
        float dense_coupling = 0.0;
        bool reverse_match = true;
        float priority = 0.0;

        CifMetas cif_metas;
        CafMetas caf_metas;

        vector<string> keypoints;
        vector<pair<int, int>> skeleton;
        vector<pair<int, int>> out_skeleton;
        vector<pair<int, int>> skeleton_m1;
        vector<float> confidence_scales;
        vector<float> score_weights;

        map<int, map<int, pair<int, bool>>> by_target;
        map<int, map<int, pair<int, bool>>> by_source;

        Keypoints nms_keypoints;

    public:
        CifCaf();
        CifCaf(
            float *field1,
            float *field2,
            unsigned H,
            unsigned W,
            const vector<float> &confidence_scales = vector<float>(),
            bool nms = true
        );

        vector<Annotation> decode(vector<Annotation> &initial_annotations = INITIAL_ANNOTATIONS);
        void annotation_inverse(vector<Annotation> &annotations, const struct PreprocessingMeta &meta);

        void _grow(Annotation &ann, CafScored &caf_scored, bool reverse_match=true);
        void mark_occupied(Annotation &ann, Occupancy &occupied);
        void _add_to_frontier(
            priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> &frontier,
            set<pair<int, int>> &in_frontier,
            Annotation &ann,
            int start_i
        );
        HeapItem _frontier_get(
            priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> &frontier,
            Annotation &ann,
            CafScored &caf_scored,
            bool reverse_match
        );
        vector<float> connection_value(
            Annotation &ann,
            CafScored &caf_scored,
            int start_i,
            int end_i,
            bool reverse_match
        );
};

#endif