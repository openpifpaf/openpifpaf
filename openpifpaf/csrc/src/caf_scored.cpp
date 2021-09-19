#include <algorithm>

#include "openpifpaf/decoder/utils/caf_scored.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double CafScored::default_score_th = 0.3;
bool CafScored::ablation_no_rescore = false;


float CafScored::cifhr_value(int64_t f, float x, float y, float default_value) {
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


void CafScored::fill(const torch::Tensor& caf_field, int64_t stride, const torch::Tensor& skeleton) {
    TORCH_CHECK(skeleton.dtype() == torch::kInt64, "skeleton must be of type LongTensor");

    auto caf_field_a = caf_field.accessor<float, 4>();
    auto skeleton_a = skeleton.accessor<int64_t, 2>();
    int64_t n_fields = caf_field_a.size(0);

    forward.resize(n_fields);
    backward.resize(n_fields);

    float c, forward_hr, backward_hr;
    for (int64_t f=0; f < n_fields; f++) {
        for (int64_t j=0; j < caf_field_a.size(2); j++) {
            for (int64_t i=0; i < caf_field_a.size(3); i++) {
                c = caf_field_a[f][1][j][i];
                if (c < score_th) continue;

                CompositeAssociation ca_forward(
                    c,
                    caf_field_a[f][2][j][i] * stride,
                    caf_field_a[f][3][j][i] * stride,
                    caf_field_a[f][4][j][i] * stride,
                    caf_field_a[f][5][j][i] * stride,
                    caf_field_a[f][6][j][i] * stride,
                    caf_field_a[f][7][j][i] * stride
                );
                CompositeAssociation ca_backward(
                    c,
                    ca_forward.x2,
                    ca_forward.y2,
                    ca_forward.x1,
                    ca_forward.y1,
                    ca_forward.s2,
                    ca_forward.s1
                );

                // rescore
                if (!ablation_no_rescore) {
                    forward_hr = cifhr_value(skeleton_a[f][1], ca_forward.x2, ca_forward.y2, 0.0);
                    backward_hr = cifhr_value(skeleton_a[f][0], ca_backward.x2, ca_backward.y2, 0.0);
                    ca_forward.c = ca_forward.c * (cif_floor + (1.0 - cif_floor) * forward_hr);
                    ca_backward.c = ca_backward.c * (cif_floor + (1.0 - cif_floor) * backward_hr);
                }

                // accumulate
                if (ca_forward.c > score_th) {
                    forward[f].push_back(ca_forward);
                }
                if (ca_backward.c > score_th) {
                    backward[f].push_back(ca_backward);
                }
            }
        }
    }
}


std::vector<torch::Tensor> to_tensors(const std::vector<std::vector<CompositeAssociation> >& vectors) {
    std::vector<torch::Tensor> tensors;

    for (const auto& associations : vectors) {
        int64_t n = associations.size();
        auto tensor = torch::from_blob(
            const_cast<void*>(reinterpret_cast<const void*>(associations.data())),
            { n, 7 }
        );
        tensors.push_back(tensor);
    }

    return tensors;
}


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor> > CafScored::get(void) {
    return { to_tensors(forward), to_tensors(backward) };
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
