#include <algorithm>

#include "openpifpaf/decoder/utils/cif_seeds.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double CifSeeds::threshold = 0.2;
double CifDetSeeds::threshold = 0.2;
bool CifSeeds::ablation_nms = false;
bool CifSeeds::ablation_no_rescore = false;


float cifhr_value(torch::TensorAccessor<float, 3UL> cifhr_a,
                  double cifhr_revision,
                  int64_t f, float x, float y, float default_value = -1.0) {
    float max_x = static_cast<float>(cifhr_a.size(2)) - 0.51;
    float max_y = static_cast<float>(cifhr_a.size(1)) - 0.51;
    if (f >= cifhr_a.size(0) || x < -0.49 || y < -0.49 || x > max_x || y > max_y) {
        return default_value;
    }

    // effectively rounding: int(float_value + 0.5)
    float value = cifhr_a[f][int64_t(y + 0.5)][int64_t(x + 0.5)] - cifhr_revision;
    if (value < 0.0) return default_value;
    return value;
}


void CifSeeds::fill(const torch::Tensor& cif_field, int64_t stride) {
    torch::optional<torch::Tensor> max_pooled;
    torch::optional<torch::TensorAccessor<float, 3>> max_pooled_a;
    if (ablation_nms) {
        auto confidence = cif_field.index({torch::indexing::Slice(), 1});
        max_pooled = torch::max_pool2d(confidence, 3, 1, 1);
        max_pooled_a = max_pooled.value().accessor<float, 3>();
    }

    auto cif_field_a = cif_field.accessor<float, 4>();
    float c, x, y, s;
    for (int64_t f=0; f < cif_field_a.size(0); f++) {
        for (int64_t j=0; j < cif_field_a.size(2); j++) {
            for (int64_t i=0; i < cif_field_a.size(3); i++) {
                c = cif_field_a[f][1][j][i];
                if (c < threshold) continue;
                if (ablation_nms) {
                    if (c < max_pooled_a.value()[f][j][i]) continue;
                }

                x = cif_field_a[f][2][j][i] * stride;
                y = cif_field_a[f][3][j][i] * stride;

                if (!ablation_no_rescore) {
                    c = 0.9 * cifhr_value(cifhr_a, cifhr_revision, f, x, y) + 0.1 * c;
                }
                if (c < threshold) continue;

                s = cif_field_a[f][4][j][i] * stride;
                seeds.push_back(Seed(f, c, x, y, s));
            }
        }
    }
}


void CifDetSeeds::fill(const torch::Tensor& cifdet_field, int64_t stride) {
    auto cif_field_a = cifdet_field.accessor<float, 4>();

    float c, v, x, y, w, h;
    for (int64_t f=0; f < cif_field_a.size(0); f++) {
        for (int64_t j=0; j < cif_field_a.size(2); j++) {
            for (int64_t i=0; i < cif_field_a.size(3); i++) {
                c = cif_field_a[f][1][j][i];
                if (c < threshold) continue;

                x = cif_field_a[f][2][j][i] * stride;
                y = cif_field_a[f][3][j][i] * stride;
                v = 0.9 * cifhr_value(cifhr_a, cifhr_revision, f, x, y) + 0.1 * c;
                if (v < threshold) continue;

                w = cif_field_a[f][4][j][i] * stride;
                h = cif_field_a[f][5][j][i] * stride;
                seeds.push_back(DetSeed(f, v, x, y, w, h));
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


std::tuple<torch::Tensor, torch::Tensor> CifDetSeeds::get(void) {
    std::sort(
        seeds.begin(), seeds.end(),
        [](const DetSeed& a, const DetSeed& b) { return a.v > b.v; }
    );
    int64_t n_seeds = seeds.size();

    auto field_tensor = torch::empty({ n_seeds }, torch::dtype(torch::kInt64));
    auto seed_tensor = torch::empty({ n_seeds, 5 });
    auto field_tensor_a = field_tensor.accessor<int64_t, 1>();
    auto seed_tensor_a = seed_tensor.accessor<float, 2>();

    for (int64_t i=0; i < n_seeds; i++) {
        field_tensor_a[i] = seeds[i].c;
        seed_tensor_a[i][0] = seeds[i].v;
        seed_tensor_a[i][1] = seeds[i].x;
        seed_tensor_a[i][2] = seeds[i].y;
        seed_tensor_a[i][3] = seeds[i].w;
        seed_tensor_a[i][4] = seeds[i].h;
    }

    return { field_tensor, seed_tensor };
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
