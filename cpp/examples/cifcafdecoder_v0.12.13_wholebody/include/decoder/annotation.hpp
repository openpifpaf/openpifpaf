#pragma once

#include "field_config.hpp"


struct DecodingStep {
    int jsi;
    int jti;
    vector<float> jsi_vec;
    vector<float> jti_vec;
};

struct AnnotationResult {
    vector<float> keypoints;
    vector<float> bbox;
    float score;
    int category_id;
};

class Annotation {
    public:
        Vector2D data;

        int category_id;
        int suppress_score_index;

        float fixed_score;
        string fixed_bbox;

        vector<float> score_weights;
        vector<float> joint_scales;
        vector<string> keypoints;
        vector<DecodingStep> decoding_order;

        vector<pair<int, int>> skeleton;
        vector<pair<int, int>> skeleton_m1;
        vector<pair<int, int>> frontier_order;

    public:
        Annotation();
        Annotation(
            const vector<string> &keypoints,
            const vector<pair<int, int>> &skeleton,
            const vector<float> score_weights,
            int category_id=1,
            int suppress_score_index=-1
        );

        Annotation add(int join_i, const vector<float> &xyv);
        vector<float> bbox_from_keypoints(const Vector2D &kps, const vector<float> &joint_scales);
        AnnotationResult json_data();
        float score();
};