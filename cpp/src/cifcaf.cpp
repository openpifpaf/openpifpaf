#include <algorithm>
#include <cmath>
#include <queue>

#include "openpifpaf/decoder/cifcaf.hpp"

#include "openpifpaf/decoder/utils/caf_scored.hpp"
#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/cif_seeds.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


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


torch::Tensor CifCaf::call(
    const torch::Tensor& cif_field,
    int64_t cif_stride,
    const torch::Tensor& caf_field,
    int64_t caf_stride
) {
    cifhr.reset(cif_field.sizes(), cif_stride);
    cifhr.accumulate(cif_field, cif_stride, 0.0, 1.0);
    auto [cifhr_accumulated, cifhr_revision] = cifhr.get_accumulated();

    utils::CifSeeds seeds(cifhr_accumulated, cifhr_revision);
    seeds.fill(cif_field, cif_stride);
    auto [seeds_f, seeds_vxys] = seeds.get();
    auto seeds_f_a = seeds_f.accessor<int64_t, 1>();
    auto seeds_vxys_a = seeds_vxys.accessor<float, 2>();

    utils::CafScored caf_scored(cifhr_accumulated, cifhr_revision, -1.0, 0.1);
    caf_scored.fill(caf_field, caf_stride, skeleton);
    auto caf_fb = caf_scored.get();

    utils::Occupancy occupied(cifhr_accumulated.sizes(), 2.0, 4.0);
    std::vector<std::vector<Joint> > annotations;

    int64_t f;
    float x, y, s;
    for (int64_t seed_i=0; seed_i < seeds_f.size(0); seed_i++) {
        f = seeds_f_a[seed_i];
        x = seeds_vxys_a[seed_i][1];
        y = seeds_vxys_a[seed_i][2];
        s = seeds_vxys_a[seed_i][3];
        if (occupied.get(f, x, y)) continue;

        std::vector<Joint> annotation(n_keypoints);
        Joint& joint = annotation[f];
        joint.v = seeds_vxys_a[seed_i][0];
        joint.x = x;
        joint.y = y;
        joint.s = s;

        _grow(annotation, caf_fb);

        for (int64_t of=0; of < n_keypoints; of++) {
            Joint& o_joint = annotation[of];
            if (o_joint.v == 0.0) continue;
            occupied.set(of, o_joint.x, o_joint.y, o_joint.s);
        }
        annotations.push_back(annotation);
    }

    auto out = torch::zeros({ int64_t(annotations.size()), n_keypoints, 4 });
    auto out_a = out.accessor<float, 3>();
    for (int64_t ann_i = 0; ann_i < int64_t(annotations.size()); ann_i++) {
        auto& ann = annotations[ann_i];
        for (int64_t joint_i = 0; joint_i < n_keypoints; joint_i++) {
            auto& joint = ann[joint_i];
            out_a[ann_i][joint_i][0] = joint.v;
            out_a[ann_i][joint_i][1] = joint.x;
            out_a[ann_i][joint_i][2] = joint.y;
            out_a[ann_i][joint_i][3] = joint.s;
        }
    }
    return out;
}


void CifCaf::_grow(
    std::vector<Joint>& ann,
    const std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor> >& caf_fb,
    bool reverse_match
) {
    assert(frontier.empty());
    assert(in_frontier.empty());

    // initialize frontier
    for (int64_t j=0; j < n_keypoints; j++) {
        if (ann[j].v == 0.0) continue;
        _frontier_add_from(ann, j);
    }

    const std::vector<torch::Tensor>& caf_forward = std::get<0>(caf_fb);
    const std::vector<torch::Tensor>& caf_backward = std::get<1>(caf_fb);

    // std::cout << frontier.top().max_score << std::endl;
}


void CifCaf::_frontier_add_from(
    std::vector<Joint>& ann,
    int64_t start_i
) {
    float max_score = sqrt(ann[start_i].v);

    for (auto&& pair : skeleton) {
        if (pair[0] == start_i) {
            if (ann[pair[1]].v > 0.0) {
                continue;
            }
            if (in_frontier.find(std::make_pair(pair[0], pair[1])) != in_frontier.end()) {
                continue;
            }
            frontier.emplace(max_score, pair[0], pair[1]);
            in_frontier.emplace(pair[0], pair[1]);
            continue;
        }
        if (pair[1] == start_i) {
            if (ann[pair[0]].v > 0.0) {
                continue;
            }
            if (in_frontier.find(std::make_pair(pair[1], pair[0])) != in_frontier.end()) {
                continue;
            }
            frontier.emplace(max_score, pair[1], pair[0]);
            in_frontier.emplace(pair[1], pair[0]);
            continue;
        }
    }
}



} // namespace decoder
} // namespace openpifpaf
