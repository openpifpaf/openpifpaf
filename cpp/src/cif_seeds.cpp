#include <algorithm>

#include "openpifpaf/decoder/utils/cif_seeds.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double CifSeeds::v_threshold = 0.2;


float CifSeeds::cifhr_value(int64_t f, float x, float y, float default_value) {
    if (x < 0.0 || y < 0.0 || x > cifhr_a.size(2) - 1 || y > cifhr_a.size(1) - 1) {
        return default_value;
    }
    return cifhr_a[f][int64_t(y)][int64_t(x)];
}



void CifSeeds::fill(const torch::Tensor& cif_field, int64_t stride) {
    auto cif_field_a = cif_field.accessor<float, 4>();

    float c, v, x, y, s;
    for (int64_t f=0; f < cif_field_a.size(0); f++) {
        for (int64_t j=0; j < cif_field_a.size(2); j++) {
            for (int64_t i=0; i < cif_field_a.size(3); i++) {
                c = cif_field_a[f][0][j][i];
                if (c < v_threshold) continue;

                x = cif_field_a[f][1][j][i] * stride;
                y = cif_field_a[f][2][j][i] * stride;
                v = 0.9 * cifhr_value(f, x, y) + 0.1 * c;
                if (v < v_threshold) continue;

                s = cif_field_a[f][4][j][i] * stride;
                seeds.push_back(Seed(f, v, x, y, s));
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> CifSeeds::get(void) {
    std::sort(
        seeds.begin(), seeds.end(),
        [](const Seed& a, const Seed& b) { return a.v > b.v; }
    );
    int64_t n_seeds = seeds.size();

    auto field_tensor = torch::empty({ n_seeds }, torch::dtype(torch::kInt64));
    auto seed_tensor = torch::empty({ n_seeds, 4 });
    auto field_tensor_a = field_tensor.accessor<int64_t, 1>();
    auto seed_tensor_a = seed_tensor.accessor<float, 2>();

    for (int64_t i=0; i < n_seeds; i++) {
        field_tensor_a[i] = seeds[i].f;
        seed_tensor_a[i][0] = seeds[i].v;
        seed_tensor_a[i][1] = seeds[i].x;
        seed_tensor_a[i][2] = seeds[i].y;
        seed_tensor_a[i][3] = seeds[i].s;
    }

    return { field_tensor, seed_tensor };
}


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
