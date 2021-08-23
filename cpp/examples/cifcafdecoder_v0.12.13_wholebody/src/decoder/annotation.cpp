#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "../../include/decoder/annotation.hpp"
#include "../../include/utils/numpy_utils.hpp"
#include "../../include/utils/vector_utils.hpp"

static const float NOTSET = -1.0;


Annotation::Annotation() {}

Annotation::Annotation(
    const vector<string> &keypoints,
    const vector<pair<int, int>> &skeleton,
    const vector<float> score_weights,
    int category_id,
    int suppress_score_index
) {
    this->keypoints = keypoints;
    this->skeleton = skeleton;
    this->score_weights = score_weights;
    this->category_id = category_id;
    this->suppress_score_index = suppress_score_index;

    int keypoints_size = (int)this->keypoints.size();
    vector<int> shape1 {keypoints_size, 3};

    this->data = vector2d_zeros(shape1);
    this->joint_scales = vector1d_zeros(keypoints_size);
    this->fixed_score = NOTSET;
    this->fixed_bbox = NOTSET;

    vector<pair<int, int>> sk = this->skeleton;
    for(size_t i = 0; i < sk.size(); i++) {
        sk[i].first -= 1;
        sk[i].second -= 1;
    }
    this->skeleton_m1 = sk;

    if(this->score_weights.empty())
        this->score_weights = vector1d_ones(keypoints_size);

    int score_weights_size = this->score_weights.size();
    if(this->suppress_score_index) {
        // TODO fix (currently never used)
        this->score_weights[score_weights_size] = 0.0;
    }

    float score_weights_sum = accumulate(this->score_weights.begin(), this->score_weights.end(), 0.0);
    std::cout << "swsum " << score_weights_sum << endl;
    for(size_t i = 0; i < this->score_weights.size(); i++) {
        this->score_weights[i] /= score_weights_sum;
    }


}

Annotation Annotation::add(int joint_i, const vector<float> &xyv) {
    this->data[joint_i] = xyv;
    return *this;
}

float Annotation::score() {
    if(this->fixed_score != -1.0)
        return this->fixed_score;

    vector<float> v;
    for(size_t i = 0; i < this->data.size(); i++) {
        v.push_back(this->data[i][2]);
    }

    if(this->suppress_score_index != -1) {
        v[this->suppress_score_index] = 0.0;
    }

    sort(v.begin(), v.end(), greater<float>());

    vector<float> r;
    for(size_t i = 0; i < v.size(); i++) {
        r.push_back(v[i] * this->score_weights[i]);
    }

    float s = 0;
    for(size_t i = 0; i < r.size(); i++) {
        s += r[i];
    }

    return s;
}

AnnotationResult Annotation::json_data() {
    vector<float> v;
    for(size_t i = 0; i < this->data.size(); i++) {
        v.push_back(this->data[i][2]);
    }
    vector<bool> v_mask = mask1d(v, 0.0);

    Vector2D keypoints = this->data;
    for(size_t i = 0; i < keypoints.size(); i++) {
        if(v_mask[i]) {
            if(keypoints[i][2] < 0.01)
                keypoints[i][2] = 0.01;
        }
    }

    vector<float> keypoints_flattened;
    for(size_t i = 0; i < keypoints.size(); i++) {
        for(size_t j = 0; j < keypoints[0].size(); j++) {
            keypoints_flattened.push_back(keypoints[i][j]);
        }
    }

    float score = max(0.001f, this->score());
    vector<float> bbox = this->bbox_from_keypoints(this->data, this->joint_scales);
    return AnnotationResult {keypoints_flattened, bbox, score, this->category_id};
}

vector<float> Annotation::bbox_from_keypoints(const Vector2D &kps, const vector<float> &joint_scales) {
    vector<float> v;
    for(size_t i = 0; i < kps.size(); i++) {
        v.push_back(kps[i][2]);
    }

    vector<bool> m = mask1d(v, 0.0);
    bool any = false;
    for(size_t i = 0; i < m.size(); i++) {
        if(m[i]) {
            any = true;
            break;
        }
    }

    if(!any) {
        return vector<float> {0.0, 0.0, 0.0, 0.0};
    }

    vector<float> kps_0_m;
    vector<float> kps_1_m;
    for(size_t i = 0; i < kps.size(); i++) {
        if(m[i]) {
            kps_0_m.push_back(kps[i][0]);
            kps_1_m.push_back(kps[i][1]);
        }
    }

    vector<float> joint_scales_m;
    for(size_t i = 0; i < joint_scales.size(); i++) {
        if(m[i]) {
            joint_scales_m.push_back(joint_scales[i]);
        }
    }

    // kps_0_m - joint_scales_m;
    vector<float> xs;
    transform(
        kps_0_m.begin(), kps_0_m.end(),
        joint_scales_m.begin(), back_inserter(xs), minus<float>()
     );
    float x = minimum1d(xs);

    vector<float> ys;
    transform(
        kps_1_m.begin(), kps_1_m.end(),
        joint_scales_m.begin(), back_inserter(ys), minus<float>()
    );
    float y = minimum1d(ys);

    vector<float> ws;
    transform(
        kps_0_m.begin(), kps_0_m.end(),
        joint_scales_m.begin(), back_inserter(ws), plus<float>()
     );
     float w = maximum1d(ws) - x;

    vector<float> hs;
    transform(
        kps_1_m.begin(), kps_1_m.end(),
        joint_scales_m.begin(), back_inserter(hs), plus<float>()
    );
    float h = maximum1d(hs) - y;

    return vector<float> {x, y, w, h};
}
