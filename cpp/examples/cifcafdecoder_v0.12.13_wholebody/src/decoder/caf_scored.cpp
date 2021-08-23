#include <algorithm>
#include <functional>
#include <iostream>
#include <cassert>

#include "../../include/decoder/caf_scored.hpp"
#include "../../include/decoder/functional.hpp"
#include "../../include/utils/numpy_utils.hpp"


CafScored::CafScored() {}

CafScored::CafScored(
    blobnd<float> &cifhr,
    unsigned H, unsigned W,
    const CafMetas &meta
) {
    this->cifhr = cifhr;
    this->meta = meta;
    this->H = H;
    this->W = W;
}

pair<Vector2D, Vector2D> CafScored::directed(int caf_i, bool forward) {
    if(forward) {
        return make_pair(this->forward[caf_i], this->backward[caf_i]);
    }

    return make_pair(this->backward[caf_i], this->forward[caf_i]);
}

Vector2D CafScored::rescore(Vector2D &nine, int joint_t) {
    if(this->cif_floor < 1.0 && joint_t < this->cifhr.shape[0]) {
        vector<float> cifhr_t = scalar_values(this->cifhr, joint_t, nine[3], nine[4], 0.0);
        float cfm = 1.0 - this->cif_floor;
        for(unsigned i = 0; i < cifhr_t.size(); i++) {
            cifhr_t[i] *= cfm;
        }
        for(unsigned i = 0; i < cifhr_t.size(); i++) {
            cifhr_t[i] += this->cif_floor;
        }
        for(unsigned i = 0; i < nine[0].size(); i++) {
            cifhr_t[i] *= nine[0][i];
        }

        nine[0] = cifhr_t;
    }

    vector<bool> mask_1d = mask1d(nine[0], this->default_score_th);
    Vector2D nine_filtered = filter_vector2D(nine, mask_1d);
    return nine_filtered;
}

void CafScored::fill_single(const Fields &all_fields) {
    float *caf = all_fields[this->meta.head_index];
    blobnd<float> caf_(caf, {this->caf_d1, this->caf_d2, this->H, this->W});

    int d1 = caf_.shape[0];
    for(int i = 0; i < d1; i++) {
        blobnd<bool> mask = mask2d(caf_, i, 0, this->default_score_th);

        bool any = false;
        for(int j = 0; j < mask.shape[0]; j++) {
            for(int k = 0; k < mask.shape[1]; k++)
                if(mask(j, k)) {
                    any = true;
                    break;
                }
        }

        if(!any) {
            vector<vector<float>> empty(9, vector<float>(0));
            this->backward.push_back(empty);
            this->forward.push_back(empty);
            continue;
        }

        Vector2D nine_filtered = filter_vector3D(caf_, i, 0, mask);

        if(this->meta.decoder_min_distance) {
            const Vector2D &nine1 = {nine_filtered[1], nine_filtered[2]};
            const Vector2D &nine2 = {nine_filtered[3], nine_filtered[4]};
            Vector2D sub = subtract2d(nine1, nine2);

            vector<float> dist = norm2d_axis0(sub);
            vector<bool> mask_dist = mask1d(dist, meta.decoder_min_distance / meta.stride);
            nine_filtered = filter_vector2D(nine_filtered, mask_dist);
        }

        if(this->meta.decoder_max_distance) {
            const Vector2D nine1 {nine_filtered[1], nine_filtered[2]};
            const Vector2D nine2 {nine_filtered[3], nine_filtered[4]};
            Vector2D sub = subtract2d(nine1, nine2);

            vector<float> dist = norm2d_axis0(sub);
            vector<bool> mask_dist = mask1d_lt(dist, this->meta.decoder_max_distance / this->meta.stride);
            nine_filtered = filter_vector2D(nine_filtered, mask_dist);
        }

        int n = nine_filtered.size();
        int m = nine_filtered[0].size();
        for(int j = 1; j < n; j++) {
            for(int k = 0; k < m; k++) {
                nine_filtered[j][k] *= this->meta.stride;
            }
        }

        vector<unsigned> nine_b_is {0, 3, 4, 1, 2, 6, 5, 8, 7};
        Vector2D nine_b;
        for(int j = 0; j < nine_b_is.size(); j++) {
            unsigned index = nine_b_is[j];
            nine_b.push_back(nine_filtered[index]);
        }
        Vector2D nine_b_rescored = this->rescore(nine_b, this->meta.head_metas.skeleton[i].first - 1);
        this->backward.push_back(nine_b_rescored);

        Vector2D nine_f = nine_filtered;
        Vector2D nine_f_rescored = this->rescore(nine_f, this->meta.head_metas.skeleton[i].second - 1);
        this->forward.push_back(nine_f_rescored);
    }
}

CafScored CafScored::fill(const Fields &all_fields) {
    this->fill_single(all_fields);

    return *this;
}