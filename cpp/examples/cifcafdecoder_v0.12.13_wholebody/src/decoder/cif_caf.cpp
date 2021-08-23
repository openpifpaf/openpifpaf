#include <cmath>

#include "../../include/decoder/caf_scored.hpp"
#include "../../include/decoder/functional.hpp"
#include "../../include/decoder/annotation.hpp"
#include "../../include/decoder/cif_seeds.hpp"
#include "../../include/decoder/occupancy.hpp"
#include "../../include/decoder/cif_caf.hpp"
#include "../../include/decoder/cif_hr.hpp"
#include "../../include/utils/vector_utils.hpp"

struct PreprocessingMeta {
    std::vector<float> offset;
    std::vector<float> scale;
    std::vector<float> valid_area;
    std::vector<int> width_height;

    bool hflip{false};
    float rotation_angle{0};
    float rotation_width{-1};
    float rotation_height{-1};
};

CifCaf::CifCaf() {}

CifCaf::CifCaf(
    float *field1,
    float *field2,
    unsigned H,
    unsigned W,
    const vector<float> &confidence_scales,
    bool nms
) {
    this->fields = Fields {field1, field2};
    this->H = H;
    this->W = W;
    this->nms = nms;

    vector<pair<int, int>> sk = this->cif_metas.head_metas.skeleton;
    for(int i = 0; i < sk.size(); i++) {
        sk[i].first -= 1;
        sk[i].second -= 1;
    }
    this->skeleton_m1 = sk;
    this->keypoints = this->cif_metas.head_metas.keypoints;
    this->score_weights = this->cif_metas.head_metas.score_weights;
    this->out_skeleton = this->caf_metas.head_metas.skeleton;
    this->confidence_scales = confidence_scales;

    this->priority += this->cif_metas.n_fields / 1000.0;
    this->priority += this->caf_metas.n_fields / 1000.0;

    for(int caf_i = 0; caf_i < this->skeleton_m1.size(); caf_i++) {
        int j1 = this->skeleton_m1[caf_i].first;
        int j2 = this->skeleton_m1[caf_i].second;
        this->by_target[j2][j1] = make_pair(caf_i, true);
        this->by_target[j1][j2] = make_pair(caf_i, false);
    }
    for(int caf_i = 0; caf_i < this->skeleton_m1.size(); caf_i++) {
        int j1 = this->skeleton_m1[caf_i].first;
        int j2 = this->skeleton_m1[caf_i].second;
        this->by_source[j1][j2] = make_pair(caf_i, true);
        this->by_source[j2][j1] = make_pair(caf_i, false);
    }
}

vector<Annotation> CifCaf::decode(vector<Annotation> &initial_annotations) {
    if(!initial_annotations.empty())
        vector<Annotation> initial_annotations;

    CifHr cif_hr(this->H, this->W, this->cif_metas);
    cif_hr.fill(this->fields);
    CifSeeds cif_seeds(
        cif_hr.accumulated,
        this->H, this->W
    );
    cif_seeds.fill(this->fields);

    CafScored caf_scored = CafScored(
        cif_hr.accumulated, this->H, this->W, this->caf_metas
    ).fill(this->fields);

    int d1 = cif_hr.accumulated.shape[0];
    int d2 = cif_hr.accumulated.shape[1];
    int d3 = cif_hr.accumulated.shape[2];
    vector<int> cif_hr_accumulated_shape {d1, d2, d3};
    Occupancy occupied = Occupancy(cif_hr_accumulated_shape, 2.0, 4.0);

    vector<Annotation> annotations;
    for(int i = 0; i < initial_annotations.size(); i++) {
        this->_grow(initial_annotations[i], caf_scored);
        annotations.push_back(initial_annotations[i]);
        this->mark_occupied(initial_annotations[i], occupied);
    }

    vector<Seed> seeds = cif_seeds.get();
    // print_vector(seeds);
    for(int i = 0; i < seeds.size(); i++) {
        float v = seeds[i].vv;
        int f = seeds[i].field;
        float x = seeds[i].xx;
        float y = seeds[i].yy;
        float s = seeds[i].ss;

        if(occupied.get(f, x, y)) {
            continue;
        }

        vector<float> xyv {x, y, v};
        Annotation ann = Annotation(this->keypoints, this->out_skeleton, this->score_weights).add(f, xyv);
        ann.joint_scales[f] = s;
        this->_grow(ann, caf_scored);
        annotations.push_back(ann);
        this->mark_occupied(ann, occupied);
    }

    if(this->force_complete) {
        // TODO implement
    }

    vector<Annotation> final_annotations;
    if(this->nms) {
        final_annotations = this->nms_keypoints.annotations(annotations);
    }

    return final_annotations;
}

void CifCaf::mark_occupied(Annotation &ann, Occupancy &occupied) {
    vector<int> joint_is;

    int rows = ann.data.size();
    for(int i = 0; i < rows; i ++) {
        if(ann.data[i][2] != 0.0)
            joint_is.push_back(i);
    }

    for(int i = 0; i < joint_is.size(); i++) {
        int joint_i = joint_is[i];
        float width = ann.joint_scales[i];

        occupied.set(joint_i, ann.data[joint_i][0], ann.data[joint_i][1], width);
    }
}

void CifCaf::_grow(Annotation &ann, CafScored &caf_scored, bool reverse_match) {
    priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> frontier;
    set<pair<int, int>> in_frontier;

    int rows = ann.data.size();
    vector<int> joint_is;
    for(int i = 0; i < rows; i ++) {
        if(ann.data[i][2] != 0.0)
            joint_is.push_back(i);
    }
    for(int joint_i = 0; joint_i < joint_is.size(); joint_i++) {
        this->_add_to_frontier(frontier, in_frontier, ann, joint_is[joint_i]);
    }

    HeapItem entry;
    while(true) {
        entry = this->_frontier_get(frontier, ann, caf_scored, reverse_match);
        if(entry.new_xysv.empty())
            break;

        vector<float> new_xysv = entry.new_xysv;
        int jsi = entry.start_i;
        int jti = entry.end_i;
        if(ann.data[jti][2] > 0.0)
            continue;

        ann.data[jti][0] = new_xysv[0];
        ann.data[jti][1] = new_xysv[1];
        ann.data[jti][2] = new_xysv[3];
        ann.joint_scales[jti] = new_xysv[2];
        struct DecodingStep decst {
            jsi, jti, ann.data[jsi], ann.data[jti]
        };
        ann.decoding_order.push_back(decst);

        this->_add_to_frontier(frontier, in_frontier, ann, jti);
    }
}


void CifCaf::_add_to_frontier(
    priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> &frontier,
    set<pair<int, int>> &in_frontier,
    Annotation &ann,
    int start_i
) {
    map<int, pair<int, bool>> by_source_items = this->by_source[start_i];

    map<int, pair<int, bool>>::iterator it;
    for(it = by_source_items.begin(); it != by_source_items.end(); it++) {
        int end_i = it->first;
        pair<int, bool> pair_ = it->second;
        int caf_i = pair_.first;

        if(ann.data[end_i][2] > 0.0)
            continue;

        if(in_frontier.count(make_pair(start_i, end_i)))
            continue;

        float max_possible_score = ann.data[start_i][2];
        if(!this->confidence_scales.empty())
            max_possible_score *= this->confidence_scales[caf_i];

        vector<float> empty;
        struct HeapItem heap_vals {-max_possible_score, empty, start_i, end_i};
        frontier.push(heap_vals);
        in_frontier.insert(make_pair(start_i, end_i));
        ann.frontier_order.push_back(make_pair(start_i, end_i));
    }
}

HeapItem CifCaf::_frontier_get(
    priority_queue<HeapItem, vector<HeapItem>, heap_item_comparer> &frontier,
    Annotation &ann,
    CafScored &caf_scored,
    bool reverse_match
) {
    if(frontier.empty()) {
        return HeapItem {-1.0, vector<float>(), -1, -1};
    }

    while(!frontier.empty()) {
        struct HeapItem entry = frontier.top();

        frontier.pop();

        if(!entry.new_xysv.empty()) {
           return entry;
        }

        int start_i = entry.start_i;
        int end_i = entry.end_i;
        if(ann.data[end_i][2] > 0.0) {
            continue;
        }

        vector<float> new_xysv = this->connection_value(ann, caf_scored, start_i, end_i, reverse_match);

        if(new_xysv[3] == 0.0)
            continue;

        float score = new_xysv[3];
        if(this->greedy)
            return HeapItem {-score, new_xysv, start_i, end_i};

        if(!this->confidence_scales.empty()) {
            pair<int, bool> pair_ = this->by_source[start_i][end_i];
            int caf_i = pair_.first;
            score *= this->confidence_scales[caf_i];
        }
        HeapItem v {-score, new_xysv, start_i, end_i};
        frontier.push(v);
    }

    if(frontier.empty()) {
        return HeapItem {-1.0, vector<float>(), -1, -1};
    }

    //TODO return something in default case to silent compiler warning
}

vector<float> CifCaf::connection_value(
    Annotation &ann,
    CafScored &caf_scored,
    int start_i,
    int end_i,
    bool reverse_match
) {
    pair<int, bool> pair_ = this->by_source[start_i][end_i];
    int caf_i = pair_.first;
    bool forward = pair_.second;

    pair<Vector2D, Vector2D> directed = caf_scored.directed(caf_i, forward);
    Vector2D caf_f = directed.first;
    Vector2D caf_b = directed.second;

    vector<float> xyv = ann.data[start_i];
    float xy_scale_s = max(0.0f, ann.joint_scales[start_i]);

    bool only_max;
    if(this->connection_method.compare("max") == 0)
        only_max = true;
    else
        only_max = false;

    vector<float> new_xysv = grow_connection_blend(caf_f, xyv[0], xyv[1], xy_scale_s, only_max);
    if(new_xysv[3] == 0.0)
        return vector<float> {0.0, 0.0, 0.0, 0.0};

    float keypoint_score = sqrt(new_xysv[3] * xyv[2]);

    if(keypoint_score < xyv[2] * this->keypoint_threshold_rel)
        return vector<float> {0.0, 0.0, 0.0, 0.0};

    float xy_scale_t = max(0.0f, new_xysv[2]);
    if(reverse_match) {
        vector<float> reverse_xyv = grow_connection_blend(caf_b, new_xysv[0], new_xysv[1], xy_scale_t, only_max);
        if(reverse_xyv[2] == 0.0)
            return vector<float> {0.0, 0.0, 0.0, 0.0};
        if(abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s)
            return vector<float> {0.0, 0.0, 0.0, 0.0};
    }

    return vector<float> {new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score};
}


void CifCaf::annotation_inverse(vector<Annotation> &annotations, const struct PreprocessingMeta &meta) {
    float angle = -meta.rotation_angle;
    float rw = meta.rotation_width;
    float rh = meta.rotation_height;

    float cangle = cos(angle / 180.0 + M_PI);
    float sangle = sin(angle / 180.0 + M_PI);

    for(int i = 0; i < annotations.size(); i++) {
        if(angle != 0.0) {
            for(int j = 0; j < annotations[i].data.size(); j++) {
                float x_old = annotations[i].data[j][0] - (rw - 1) / 2;
                float y_old = annotations[i].data[j][1] - (rh - 1) / 2;

                annotations[i].data[j][0] = (rw - 1) / 2 + cangle * x_old + sangle * y_old;
                annotations[i].data[j][1] = (rh - 1) / 2 - sangle * x_old + cangle * y_old;
            }
        }

        for(int j = 0; j < annotations[i].data.size(); j++) {
            annotations[i].data[j][0] += meta.offset[0];
            annotations[i].data[j][1] += meta.offset[1];
        }

        for(int j = 0; j < annotations[i].data.size(); j++) {
            annotations[i].data[j][0] /= meta.scale[0];
            annotations[i].data[j][1] /= meta.scale[1];
        }

        for(int j = 0; j < annotations[i].joint_scales.size(); j++) {
            annotations[i].joint_scales[j] /= meta.scale[0];
        }

        if(meta.hflip) {
            int w = meta.width_height[0];
            for(int j = 0; j < annotations[i].data.size(); j++) {
                annotations[i].data[j][0] = -annotations[i].data[j][0] + (w - 1);
            }
        }

        for(int j = 0; j < annotations[i].decoding_order.size(); j++) {
            annotations[i].decoding_order[j].jsi_vec[0] += meta.offset[0];
            annotations[i].decoding_order[j].jsi_vec[1] += meta.offset[1];

            annotations[i].decoding_order[j].jti_vec[0] += meta.offset[0];
            annotations[i].decoding_order[j].jti_vec[1] += meta.offset[1];

            annotations[i].decoding_order[j].jsi_vec[0] /= meta.scale[0];
            annotations[i].decoding_order[j].jsi_vec[1] /= meta.scale[1];

            annotations[i].decoding_order[j].jti_vec[0] /= meta.scale[0];
            annotations[i].decoding_order[j].jti_vec[1] /= meta.scale[1];
        }
    }
}
