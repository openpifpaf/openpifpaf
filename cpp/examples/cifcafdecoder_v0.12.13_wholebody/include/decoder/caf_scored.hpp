#ifndef CAF_SCORED_HPP
#define CAF_SCORED_HPP

#include <limits>

#include "field_config.hpp"


class CafScored {
    public:
        CafMetas meta;

        blobnd<float> cifhr;
        Vector3D forward;
        Vector3D backward;

        float default_score_th = 0.2;
        float cif_floor = 0.1;
        unsigned caf_d1 = 160;
        unsigned caf_d2 = 9;
        unsigned H;
        unsigned W;

    public:
        CafScored();
        CafScored(
            blobnd<float> &cifhr,
            unsigned H, unsigned W,
            const CafMetas &meta
        );

        pair<Vector2D, Vector2D> directed(int caf_i, bool forward);
        Vector2D rescore(Vector2D &nine, int joint_t);
        void fill_single(const Fields &all_fields);
        CafScored fill(const Fields &all_fields);
};

#endif