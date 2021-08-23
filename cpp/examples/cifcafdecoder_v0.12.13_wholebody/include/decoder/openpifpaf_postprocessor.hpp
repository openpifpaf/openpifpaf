#ifndef OPENPIFPAF_POSTPROCESSOR_HPP
#define OPENPIFPAF_POSTPROCESSOR_HPP

#include "field_config.hpp"
#include "cif_hr.hpp"
#include "cif_seeds.hpp"
#include "caf_scored.hpp"
#include "occupancy.hpp"
#include "cif_caf.hpp"


class OpenpifpafPostprocessor {
    public:
        float     *field1;
        float     *field2;
        unsigned  H;
        unsigned  W;
        bool      PARAM_FORCE_COMPLETE_POSE = false;
        bool      PARAM_DENSE_CONNECTIONS   = false;
        bool      PARAM_MULTI_SCALE_HFLIP   = true;
        bool      PARAM_MULTI_SCALE         = false;
        bool      PARAM_CAF_SEEDS           = false;
        bool      PARAM_GREEDY              = false;
        float     PARAM_KEYPOINT_THRESHOLD  = -1.0;
        float     PARAM_INSTANCE_THRESHOLD  = 0.1;
        float     PARAM_DENSE_COUPLING      = 0.01;
        float     PARAM_SEED_THRESHOLD      = 0.5;
        float     PARAM_CIF_THRESHOLD       = 0.1;
        float     PARAM_CAF_THRESHOLD       = 0.1;
        string    PARAM_CONNECTION_METHOD   = "blend";
        bool      PARAM_NMS                 = true;

        vector<Annotation> annotations;
        CifHr     cif_hr;
        CifSeeds  cif_seeds;
        CafScored caf_scored;
        CifCaf    cif_caf;

    public:
        OpenpifpafPostprocessor(
            float                                  *field1,
            float                                  *field2,
            unsigned                                     H,
            unsigned                                     W,
            const bool   param_force_complete_pose = false,
            const bool   param_dense_connections   = false,
            const bool   param_multi_scale_hflip   = true,
            const bool   param_multi_scale         = false,
            const bool   param_caf_seeds           = false,
            const bool   param_greedy              = false,
            float        param_keypoint_threshold  = -1.0,
            const float  param_instance_threshold  = 0.1,
            const float  param_dense_coupling      = 0.01,
            const float  param_seed_threshold      = 0.5,
            const float  param_cif_threshold       = 0.1,
            const float  param_caf_threshold       = 0.1,
            const string PARAM_CONNECTION_METHOD   = "blend",
            const bool   param_nms                  = true
        );
        OpenpifpafPostprocessor(
            vector<float>                           &field1,
            vector<float>                           &field2,
            unsigned                                     H,
            unsigned                                     W,
            const bool   param_force_complete_pose = false,
            const bool   param_dense_connections   = false,
            const bool   param_multi_scale_hflip   = true,
            const bool   param_multi_scale         = false,
            const bool   param_caf_seeds           = false,
            const bool   param_greedy              = false,
            float        param_keypoint_threshold  = -1.0,
            const float  param_instance_threshold  = 0.1,
            const float  param_dense_coupling      = 0.01,
            const float  param_seed_threshold      = 0.5,
            const float  param_cif_threshold       = 0.1,
            const float  param_caf_threshold       = 0.1,
            const string PARAM_CONNECTION_METHOD   = "blend",
            const bool   param_nms                 = true
        );

        vector<Annotation> process();
        string json_predictions();
        vector<Annotation> annotation_inverse(const struct PreprocessingMeta &meta);

};

#endif