#include <algorithm>

#include "openpifpaf/decoder/utils/caf_scored.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double CafScored::default_score_th = 0.2;


float CafScored::cifhr_value(int64_t f, float x, float y, float default_value) {
    if (x < 0.0 || y < 0.0 || x > cifhr_a.size(2) - 1 || y > cifhr_a.size(1) - 1) {
        return default_value;
    }
    return cifhr_a[f][int64_t(y)][int64_t(x)];
}



void CafScored::fill(const torch::Tensor& caf_field, int64_t stride) {
    auto caf_field_a = caf_field.accessor<float, 4>();

    float c, forward_hr, backward_hr;
    for (int64_t f=0; f < caf_field_a.size(0); f++) {
        for (int64_t j=0; j < caf_field_a.size(2); j++) {
            for (int64_t i=0; i < caf_field_a.size(3); i++) {
                c = caf_field_a[f][0][j][i];
                if (c < score_th) continue;

                auto ca_forward = CompositeAssociation(
                    c,
                    caf_field_a[f][1][j][i] * stride,
                    caf_field_a[f][2][j][i] * stride,
                    caf_field_a[f][3][j][i] * stride,
                    caf_field_a[f][4][j][i] * stride,
                    caf_field_a[f][5][j][i] * stride,
                    caf_field_a[f][6][j][i] * stride,
                    caf_field_a[f][7][j][i] * stride,
                    caf_field_a[f][8][j][i] * stride
                );
                auto ca_backward = CompositeAssociation(
                    c,
                    ca_forward.x2,
                    ca_forward.y2,
                    ca_forward.b2,
                    ca_forward.s2,
                    ca_forward.x1,
                    ca_forward.y1,
                    ca_forward.b1,
                    ca_forward.s1
                );

                // rescore
                forward_hr = cifhr_value(f, ca_forward.x2, ca_forward.y2, 0.0);
                backward_hr = cifhr_value(f, ca_backward.x2, ca_backward.y2, 0.0);
                ca_forward.c = ca_forward.c * (cif_floor + (1.0 - cif_floor) * forward_hr);
                ca_backward.c = ca_backward.c * (cif_floor + (1.0 - cif_floor) * backward_hr);

                // accumulate
                if (ca_forward.c > score_th) {
                    forward.push_back(ca_forward);
                }
                if (ca_backward.c > score_th) {
                    backward.push_back(ca_backward);
                }
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> CafScored::get(void) {
    int64_t n_forward = forward.size();
    int64_t n_backward = backward.size();

    auto forward_tensor = torch::empty({ n_forward, 9 });
    auto backward_tensor = torch::empty({ n_backward, 9 });
    auto forward_tensor_a = forward_tensor.accessor<float, 2>();
    auto backward_tensor_a = backward_tensor.accessor<float, 2>();

    int64_t i = 0;
    for (auto& ca : forward) {
        forward_tensor_a[i][0] = ca.c;
        i++;
    }

    int64_t j = 0;
    for (auto& ca : backward) {
        backward_tensor_a[j][0] = ca.c;
        j++;
    }

    return { forward_tensor, backward_tensor };
}


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
