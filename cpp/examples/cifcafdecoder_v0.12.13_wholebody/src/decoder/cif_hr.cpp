#include <functional>
#include <algorithm>

#include "../../include/decoder/functional.hpp"
#include "../../include/utils/numpy_utils.hpp"
#include "../../include/decoder/cif_hr.hpp"


CifHr::CifHr() {}

CifHr::CifHr(unsigned H, unsigned W, const CifMetas &meta) {
    this->meta = meta;
    this->H = H;
    this->W = W;
}

void CifHr::fill_single(const Fields &all_fields) {
    return this->fill(all_fields);
}

void CifHr::fill(const Fields &all_fields) {
    blobnd<float> cifs_(all_fields[this->meta.head_index], {this->cif_d1, this->cif_d2, this->H, this->W});

    unsigned d1 = cifs_.shape[0];
    unsigned d3 = cifs_.shape[2];
    unsigned d4 = cifs_.shape[3];

    vector<unsigned> shape;
    if(this->accumulated.size == 0) {
        shape = {
            (unsigned)d1,
            (unsigned)((d3 - 1) * this->meta.stride + 1),
            (unsigned)((d4 - 1) * this->meta.stride + 1)
        };
    } else {
        unsigned ad1 = this->accumulated.shape[0];
        unsigned ad2 = this->accumulated.shape[1];
        unsigned ad3 = this->accumulated.shape[2];
        shape = {ad1, ad2, ad3};
    }
    blobnd<float> ta({shape[0], shape[1], shape[2]});

    int m = cifs_.shape[0];
    for(int j = 0; j < m; j++) {
        this->accumulate(1, ta, j, cifs_, j, this->meta.stride, this->meta.decoder_min_scale);
    }
    if(this->accumulated.size == 0) {
        this->accumulated = ta;
    } else {
        // TODO fix case (if needed)
        // this->accumulated = maximum3d(ta, this->accumulated);
    }
}

void CifHr::accumulate(
    int len_cifs,
    blobnd<float> &t, int t_index,
    blobnd<float> &p, int p_index,
    int stride,
    float min_scale
) {
    blobnd<bool> mask = mask2d(p, p_index, 0, this->v_threshold);
    Vector2D p_filtered = filter_vector3D(p, p_index, 0, mask);

    if(min_scale)
         p_filtered = filter_vector2D(p_filtered, 4, min_scale / stride);

    vector<float> v = p_filtered[0];
    vector<float> x = p_filtered[1];
    vector<float> y = p_filtered[2];
    vector<float> scale = p_filtered[4];

    // v = v / this->neighbors / len_cifs
    transform(
        v.begin(), v.end(), v.begin(),
        bind(divides<float>(), placeholders::_1, this->neighbors)
    );
    transform(
        v.begin(), v.end(), v.begin(),
        bind(divides<float>(), placeholders::_1, len_cifs)
    );
    // x = x * stride
    transform(
        x.begin(), x.end(), x.begin(),
        bind(multiplies<float>(), placeholders::_1, stride)
    );
    // y = y * stride
    transform(
        y.begin(), y.end(), y.begin(),
        bind(multiplies<float>(), placeholders::_1, stride)
    );
    // scale = scale * 0.5 * stride
    transform(
        scale.begin(), scale.end(), scale.begin(),
        bind(multiplies<float>(), placeholders::_1, 0.5 * stride)
    );
    vector<float> sigma = maximum(1.0, scale);

    scalar_square_add_gauss_with_max(
        t, t_index,
        x, y, sigma, v, 1.0
    );
}
