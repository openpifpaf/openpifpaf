#include <functional>
#include <algorithm>

#include "../../include/decoder/functional.hpp"
#include "../../include/decoder/cif_seeds.hpp"
#include "../../include/utils/numpy_utils.hpp"


bool compare_seeds(const struct Seed& s1, const struct Seed& s2) {
    if (s1.vv != s2.vv) {
        return s1.vv > s2.vv;
    } else if (s1.field != s2.field) {
        return s1.field > s2.field;
    } else if (s1.xx != s2.xx) {
        return s1.xx > s2.xx;
    } else if (s1.yy != s2.yy) {
        return s1.yy > s2.yy;
    } else if (s1.ss != s2.ss) {
        return s1.ss > s2.ss;
    }

    return false;
}

CifSeeds::CifSeeds() {}

CifSeeds::CifSeeds(blobnd<float> &cifhr, unsigned H, unsigned W) {
    this->H = H;
    this->W = W;
    this->cifhr = cifhr;
}

void CifSeeds::fill_cif(float *cif, int stride, float min_scale, const vector<int> &seed_mask) {
    blobnd<float> cif_(cif, {this->cif_d1, this->cif_d2, this->H, this->W});
    // std::cout << "dims " << this->cif_d1 << " " << this->cif_d2 << " " << this->H << " " << this->W << endl;
    // std::cout << cif_.size << endl;
    Vector2D p_filtered;
    vector<bool> m;

    float sv = 0.0;
    int d1 = cif_.shape[0];
    for(int i = 0; i < d1; i++) {
        if(!seed_mask.empty() && !seed_mask[i]) {
            continue;
        }
        blobnd<bool> mask = mask2d(cif_, i, 0, this->threshold);
        p_filtered = filter_vector3D(cif_, i, 0, mask);
        if(min_scale) {
            p_filtered = filter_vector2D(p_filtered, 4, min_scale / stride);
        }

        vector<float> c = p_filtered[0];
        vector<float> x = p_filtered[1];
        vector<float> y = p_filtered[2];
        vector<float> s = p_filtered[4];
        // x = x * stirde
        vector<float> x_stride;
        transform(
            x.begin(), x.end(), back_inserter(x_stride),
            bind(multiplies<float>(), placeholders::_1, stride)
        );
        // y = y * stride
        vector<float> y_stride;
        transform(
            y.begin(), y.end(), back_inserter(y_stride),
            bind(multiplies<float>(), placeholders::_1, stride)
        );

        vector<float> v = scalar_values(this->cifhr, i, x_stride, y_stride, 0.0, 0);
        // v = 0.9 * v + 0.1 * c
        transform(
            v.begin(), v.end(), v.begin(),
            bind(multiplies<float>(), placeholders::_1, 0.9)
        );
        transform(
            c.begin(), c.end(), c.begin(),
            bind(multiplies<float>(), placeholders::_1, 0.1)
        );
        transform(
            v.begin(), v.end(),
            c.begin(), v.begin(),
            plus<float>()
        );

        if(this->score_scale != 1.0) {
            // v = v * score_scale
            transform(
                v.begin(), v.end(), v.begin(),
                bind(multiplies<float>(), placeholders::_1, 0.9)
            );
        }

        m = mask1d(v, this->threshold);
        // x = x[m] * stride
        x = filter1d(x, m);
        transform(
            x.begin(), x.end(), x.begin(),
            bind(multiplies<float>(), placeholders::_1, stride)
        );

        // y = y[m] * stride
        y = filter1d(y, m);
        transform(
            y.begin(), y.end(), y.begin(),
            bind(multiplies<float>(), placeholders::_1, stride)
        );
        // v = v[m]
        v = filter1d(v, m);
        // s = s[m] * stride
        s = filter1d(s, m);
        transform(
            s.begin(), s.end(), s.begin(),
            bind(multiplies<float>(), placeholders::_1, stride)
        );

        for(int j = 0; j < x.size(); j++) {
            float vv = v[j];
            float xx = x[j];
            float yy = y[j];
            float ss = s[j];
            struct Seed seed {vv, i, xx, yy, ss};
            this->seeds.push_back(seed);
        }
    }
}

vector<Seed> CifSeeds::get() {
    sort(this->seeds.begin(), this->seeds.end(), compare_seeds);
    return this->seeds;
}

void CifSeeds::fill(const Fields &fields) {
    const vector<int> seed_mask;
    float min_scale = 0.0;
    int cif_i = 0;
    int stride = 8;

    this->fill_cif(fields[cif_i], stride, min_scale, seed_mask);
}
