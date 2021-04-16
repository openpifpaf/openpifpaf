#include <algorithm>
#include <cmath>

#include "openpifpaf/decoder/cifcaf.hpp"


namespace openpifpaf {
namespace decoder {


std::vector<double> grow_connection_blend(const torch::Tensor& caf, double x, double y, double xy_scale, bool only_max) {
    /*
    Blending the top two candidates with a weighted average.

    Similar to the post processing step in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs".
    */
    auto caf_a = caf.accessor<float, 2>();
    float sigma_filter = 2.0 * xy_scale;  // 2.0 = 4 sigma
    float sigma2 = 0.25 * xy_scale * xy_scale;
    float d2, score;

    int64_t score_1_i = 0, score_2_i = 0;
    float score_1 = 0.0, score_2 = 0.0;
    for (int64_t i=0; i < caf_a.size(0); i++) {
        if (caf_a[i][1] < x - sigma_filter) continue;
        if (caf_a[i][1] > x + sigma_filter) continue;
        if (caf_a[i][2] < y - sigma_filter) continue;
        if (caf_a[i][2] > y + sigma_filter) continue;

        // source distance
        d2 = std::pow(caf_a[i][1] - x, 2) + std::pow(caf_a[i][2] - y, 2);

        // combined value and source distance
        score = std::exp(-0.5 * d2 / sigma2) * caf_a[i][0];

        if (score >= score_1) {  // if score is equal to score_1, make sure score_2 is filled
            score_2_i = score_1_i;
            score_2 = score_1;
            score_1_i = i;
            score_1 = score;
        } else if (score > score_2) {
            score_2_i = i;
            score_2 = score;
        }
    }

    if (score_1 == 0.0) return { 0, 0, 0, 0 };

    float entry_1[4] = {  // xybs
        caf_a[score_1_i][3], caf_a[score_1_i][4],
        caf_a[score_1_i][6], caf_a[score_1_i][8]
    };
    if (only_max)
        return { entry_1[0], entry_1[1], entry_1[3], score_1 };
    if (score_2 < 0.01 || score_2 < 0.5 * score_1)
        return { entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5 };

    // blend
    float entry_2[4] = {  // xybs
        caf_a[score_2_i][3], caf_a[score_2_i][4],
        caf_a[score_2_i][6], caf_a[score_2_i][8]
    };

    float blend_d2 = std::pow(entry_1[0] - entry_2[0], 2) + std::pow(entry_1[1] - entry_2[1], 2);
    if (blend_d2 > std::pow(entry_1[3], 2) / 4.0)
        return { entry_1[0], entry_1[1], entry_1[3], score_1 * 0.5 };

    return {  // xysv
        (score_1 * entry_1[0] + score_2 * entry_2[0]) / (score_1 + score_2),
        (score_1 * entry_1[1] + score_2 * entry_2[1]) / (score_1 + score_2),
        (score_1 * entry_1[3] + score_2 * entry_2[3]) / (score_1 + score_2),
        0.5 * (score_1 + score_2)
    };
}


} // namespace decoder
} // namespace openpifpaf
