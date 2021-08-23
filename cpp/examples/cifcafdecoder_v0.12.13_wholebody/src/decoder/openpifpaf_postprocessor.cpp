#include <cassert>

#include "../../include/decoder/openpifpaf_postprocessor.hpp"
#include "../../deps/jsonl/json.hpp"

using namespace nlohmann;


OpenpifpafPostprocessor::OpenpifpafPostprocessor(
        float        *field1,
        float        *field2,
        unsigned     H,
        unsigned     W,
        const bool   param_force_complete_pose,
        const bool   param_dense_connections,
        const bool   param_multi_scale_hflip,
        const bool   param_multi_scale,
        const bool   param_caf_seeds,
        const bool   param_greedy,
        float        param_keypoint_threshold,
        const float  param_instance_threshold,
        const float  param_dense_coupling,
        const float  param_seed_threshold,
        const float  param_cif_threshold,
        const float  param_caf_threshold,
        const string param_connection_method,
        const bool   param_nms
) {
    this->field1                    = field1;
    this->field2                    = field2;
    this->H                         = H;
    this->W                         = W;
    this->PARAM_FORCE_COMPLETE_POSE = param_force_complete_pose;
    this->PARAM_DENSE_CONNECTIONS   = param_dense_connections;
    this->PARAM_MULTI_SCALE_HFLIP   = param_multi_scale_hflip;
    this->PARAM_MULTI_SCALE         = param_multi_scale;
    this->PARAM_CAF_SEEDS           = param_caf_seeds;
    this->PARAM_GREEDY              = param_greedy;
    this->PARAM_KEYPOINT_THRESHOLD  = param_keypoint_threshold;
    this->PARAM_INSTANCE_THRESHOLD  = param_instance_threshold;
    this->PARAM_DENSE_COUPLING      = param_dense_coupling;
    this->PARAM_SEED_THRESHOLD      = param_seed_threshold;
    this->PARAM_CIF_THRESHOLD       = param_cif_threshold;
    this->PARAM_CAF_THRESHOLD       = param_caf_threshold;
    this->PARAM_CONNECTION_METHOD   = param_connection_method;
    this->PARAM_NMS                 = param_nms;

    this->cif_caf = CifCaf(
        this->field1,
        this->field2,
        this->H,
        this->W
    );
}

OpenpifpafPostprocessor::OpenpifpafPostprocessor(
    vector<float> &field1,
    vector<float> &field2,
    unsigned     H,
    unsigned     W,
    const bool   param_force_complete_pose,
    const bool   param_dense_connections,
    const bool   param_multi_scale_hflip,
    const bool   param_multi_scale,
    const bool   param_caf_seeds,
    const bool   param_greedy,
    float        param_keypoint_threshold,
    const float  param_instance_threshold,
    const float  param_dense_coupling,
    const float  param_seed_threshold,
    const float  param_cif_threshold,
    const float  param_caf_threshold,
    const string param_connection_method,
    const bool   param_nms
) {
    this->field1                    = new float[field1.size()];
    std::copy(field1.data(), field1.data() + field1.size(), this->field1);
    this->field2                    = new float[field2.size()];
    std::copy(field2.data(), field2.data() + field2.size(), this->field2);

    this->H                         = H;
    this->W                         = W;
    this->PARAM_FORCE_COMPLETE_POSE = param_force_complete_pose;
    this->PARAM_DENSE_CONNECTIONS   = param_dense_connections;
    this->PARAM_MULTI_SCALE_HFLIP   = param_multi_scale_hflip;
    this->PARAM_MULTI_SCALE         = param_multi_scale;
    this->PARAM_CAF_SEEDS           = param_caf_seeds;
    this->PARAM_GREEDY              = param_greedy;
    this->PARAM_KEYPOINT_THRESHOLD  = param_keypoint_threshold;
    this->PARAM_INSTANCE_THRESHOLD  = param_instance_threshold;
    this->PARAM_DENSE_COUPLING      = param_dense_coupling;
    this->PARAM_SEED_THRESHOLD      = param_seed_threshold;
    this->PARAM_CIF_THRESHOLD       = param_cif_threshold;
    this->PARAM_CAF_THRESHOLD       = param_caf_threshold;
    this->PARAM_CONNECTION_METHOD   = param_connection_method;
    this->PARAM_NMS                 = param_nms;

    this->cif_caf = CifCaf(
        this->field1, this->field2,
        this->H,
        this->W
    );
}


vector<Annotation> OpenpifpafPostprocessor::process() {
    vector<Annotation> annotations = this->cif_caf.decode();
    this->annotations = annotations;

    return annotations;
}

string OpenpifpafPostprocessor::json_predictions() {
    vector<Annotation> annotations = this->cif_caf.decode();
    this->annotations = annotations;

    json result;
    for(unsigned i = 0; i < annotations.size(); i++) {
        AnnotationResult ar = annotations[i].json_data();
        result.push_back({
            {"keypoints", ar.keypoints},
            {"bbox", ar.bbox},
            {"score", ar.score}
        });
    }
    return result.dump();
}

vector<Annotation> OpenpifpafPostprocessor::annotation_inverse(const struct PreprocessingMeta &meta) {
    this->cif_caf.annotation_inverse(this->annotations, meta);

    return this->annotations;
}