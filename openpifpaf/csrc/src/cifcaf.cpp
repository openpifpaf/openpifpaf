#include <algorithm>
#include <cmath>
#include <queue>

#include "openpifpaf/decoder/cifcaf.hpp"

#include "openpifpaf/decoder/utils/caf_scored.hpp"
#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/cif_seeds.hpp"
#include "openpifpaf/decoder/utils/nms.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {


void test_op_int64(int64_t v) { std::cout << v << std::endl; }
void test_op_double(double v) { std::cout << v << std::endl; }



bool CifCaf::greedy = false;
double CifCaf::keypoint_threshold = 0.15;
double CifCaf::keypoint_threshold_rel = 0.5;
bool CifCaf::reverse_match = true;
bool CifCaf::force_complete = false;
double CifCaf::force_complete_caf_th = 0.001;


Joint grow_connection_blend(const torch::Tensor& caf, double x, double y, double xy_scale, bool only_max) {
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
        caf_a[score_1_i][6], fmax(0.0f, caf_a[score_1_i][8])
    };
    if (only_max)
        return { score_1, entry_1[0], entry_1[1], entry_1[3] };
    if (score_2 < 0.01 || score_2 < 0.5 * score_1)
        return { 0.5 * score_1, entry_1[0], entry_1[1], entry_1[3] };

    // blend
    float entry_2[4] = {  // xybs
        caf_a[score_2_i][3], caf_a[score_2_i][4],
        caf_a[score_2_i][6], fmax(0.0f, caf_a[score_2_i][8])
    };

    float blend_d2 = std::pow(entry_1[0] - entry_2[0], 2) + std::pow(entry_1[1] - entry_2[1], 2);
    if (blend_d2 > std::pow(entry_1[3], 2) / 4.0)
        return { 0.5 * score_1, entry_1[0], entry_1[1], entry_1[3] };

    return {
        0.5 * (score_1 + score_2),
        (score_1 * entry_1[0] + score_2 * entry_2[0]) / (score_1 + score_2),
        (score_1 * entry_1[1] + score_2 * entry_2[1]) / (score_1 + score_2),
        (score_1 * entry_1[3] + score_2 * entry_2[3]) / (score_1 + score_2)
    };
}

std::vector<double> grow_connection_blend_py(const torch::Tensor& caf, double x, double y, double s, bool only_max) {
    Joint j(grow_connection_blend(caf, x, y, s, only_max));
    return { j.x, j.y, j.s, j.v };  // xysv
}


torch::Tensor cifcaf_op(
    int64_t n_keypoints,
    const torch::Tensor& skeleton,
    const torch::Tensor& cif_field,
    int64_t cif_stride,
    const torch::Tensor& caf_field,
    int64_t caf_stride
) {
    return CifCaf(n_keypoints, skeleton).call(cif_field, cif_stride, caf_field, caf_stride);
}


torch::Tensor CifCaf::call(
    const torch::Tensor& cif_field,
    int64_t cif_stride,
    const torch::Tensor& caf_field,
    int64_t caf_stride
) {
    TORCH_CHECK(cif_field.device().is_cpu(), "cif_field must be a CPU tensor");
    TORCH_CHECK(caf_field.device().is_cpu(), "caf_field must be a CPU tensor");

    cifhr.reset(cif_field.sizes(), cif_stride);
    cifhr.accumulate(cif_field, cif_stride, 0.0, 1.0);
    auto [cifhr_accumulated, cifhr_revision] = cifhr.get_accumulated();

    utils::CifSeeds seeds(cifhr_accumulated, cifhr_revision);
    seeds.fill(cif_field, cif_stride);
    auto [seeds_f, seeds_vxys] = seeds.get();
    auto seeds_f_a = seeds_f.accessor<int64_t, 1>();
    auto seeds_vxys_a = seeds_vxys.accessor<float, 2>();
    // std::cout << "seeds: " << seeds_f_a.size(0) << std::endl;

    utils::CafScored caf_scored(cifhr_accumulated, cifhr_revision, -1.0, 0.1);
    caf_scored.fill(caf_field, caf_stride, skeleton);
    auto caf_fb = caf_scored.get();
    // auto caf_f = std::get<0>(caf_fb);
    // size_t n_caf_f = std::accumulate(caf_f.begin(), caf_f.end(), 0,
    //                                  [](size_t a, torch::Tensor& b) { return a + b.size(0); });
    // auto caf_b = std::get<1>(caf_fb);
    // size_t n_caf_b = std::accumulate(caf_b.begin(), caf_b.end(), 0,
    //                                  [](size_t a, torch::Tensor& b) { return a + b.size(0); });
    // std::cout << "caf forward: " << n_caf_f << ", caf backward: " << n_caf_b << std::endl;

    occupancy.reset(cifhr_accumulated.sizes());
    std::vector<std::vector<Joint> > annotations;

    int64_t f;
    float x, y, s;
    for (int64_t seed_i=0; seed_i < seeds_f.size(0); seed_i++) {
        f = seeds_f_a[seed_i];
        x = seeds_vxys_a[seed_i][1];
        y = seeds_vxys_a[seed_i][2];
        s = seeds_vxys_a[seed_i][3];
        if (occupancy.get(f, x, y)) continue;

        std::vector<Joint> annotation(n_keypoints);
        Joint& joint = annotation[f];
        joint.v = seeds_vxys_a[seed_i][0];
        joint.x = x;
        joint.y = y;
        joint.s = s;

        _grow(&annotation, caf_fb);

        for (int64_t of=0; of < n_keypoints; of++) {
            Joint& o_joint = annotation[of];
            if (o_joint.v == 0.0) continue;
            occupancy.set(of, o_joint.x, o_joint.y, o_joint.s);
        }
        annotations.push_back(annotation);
    }

    if (force_complete) {
        _force_complete(&annotations, cifhr_accumulated, cifhr_revision, caf_field, caf_stride);
        for (auto&& ann : annotations) _flood_fill(&ann);
    }

    utils::NMSKeypoints().call(&occupancy, &annotations);

    auto out = torch::zeros({ int64_t(annotations.size()), n_keypoints, 4 });
    auto out_a = out.accessor<float, 3>();
    for (int64_t ann_i = 0; ann_i < int64_t(annotations.size()); ann_i++) {
        auto& ann = annotations[ann_i];
        for (int64_t joint_i = 0; joint_i < n_keypoints; joint_i++) {
            Joint& joint = ann[joint_i];
            out_a[ann_i][joint_i][0] = joint.v;
            out_a[ann_i][joint_i][1] = joint.x;
            out_a[ann_i][joint_i][2] = joint.y;
            out_a[ann_i][joint_i][3] = joint.s;
        }
    }
    return out;
}


void CifCaf::_grow(
    std::vector<Joint>* ann,
    const caf_fb_t& caf_fb,
    bool reverse_match
) {
    while (!frontier.empty()) frontier.pop();
    in_frontier.clear();

    // initialize frontier
    for (int64_t j=0; j < n_keypoints; j++) {
        if ((*ann)[j].v == 0.0) continue;
        _frontier_add_from(*ann, j);
    }

    while (!frontier.empty()) {
        FrontierEntry entry(frontier.top());
        frontier.pop();
        // Was the target already filled by something else?
        if ((*ann)[entry.end_i].v > 0.0) continue;

        // Is entry not fully computed?
        if (entry.joint.v == 0.0) {
            Joint new_joint = _connection_value(
                *ann, caf_fb, entry.start_i, entry.end_i, reverse_match);
            if (new_joint.v == 0.0) continue;

            if (!greedy) {
                // if self.confidence_scales is not None:
                //     caf_i, _ = self.by_source[start_i][end_i]
                //     score = score * self.confidence_scales[caf_i]
                frontier.emplace(new_joint.v, new_joint, entry.start_i, entry.end_i);
                continue;
            }

            entry.max_score = new_joint.v;
            entry.joint = new_joint;
        }

        (*ann)[entry.end_i] = entry.joint;
        _frontier_add_from(*ann, entry.end_i);
    }
}


void CifCaf::_frontier_add_from(
    const std::vector<Joint>& ann,
    int64_t start_i
) {
    float max_score = sqrt(ann[start_i].v);

    auto skeleton_a = skeleton.accessor<int64_t, 2>();
    for (int64_t f=0; f < skeleton_a.size(0); f++) {
        int64_t pair_0 = skeleton_a[f][0];
        int64_t pair_1 = skeleton_a[f][1];

        if (pair_0 == start_i) {
            if (ann[pair_1].v > 0.0) continue;
            if (in_frontier.find(std::make_pair(pair_0, pair_1)) != in_frontier.end()) {
                continue;
            }
            frontier.emplace(max_score, pair_0, pair_1);
            in_frontier.emplace(pair_0, pair_1);
            continue;
        }
        if (pair_1 == start_i) {
            if (ann[pair_0].v > 0.0) continue;
            if (in_frontier.find(std::make_pair(pair_1, pair_0)) != in_frontier.end()) {
                continue;
            }
            frontier.emplace(max_score, pair_1, pair_0);
            in_frontier.emplace(pair_1, pair_0);
            continue;
        }
    }
}


Joint CifCaf::_connection_value(
    const std::vector<Joint>& ann,
    const caf_fb_t& caf_fb,
    int64_t start_i,
    int64_t end_i,
    bool reverse_match_
) {
    int64_t caf_i = 0;
    bool forward = true;

    auto skeleton_a = skeleton.accessor<int64_t, 2>();
    for (int64_t f=0; f < skeleton_a.size(0); f++) {
        int64_t pair_0 = skeleton_a[f][0];
        int64_t pair_1 = skeleton_a[f][1];

        if (pair_0 == start_i && pair_1 == end_i) {
            forward = true;
            break;
        }
        if (pair_1 == start_i && pair_0 == end_i) {
            forward = false;
            break;
        }
        caf_i++;
    }
    assert(caf_i < skeleton_a.size(0));
    auto caf_f = forward ? std::get<0>(caf_fb)[caf_i] : std::get<1>(caf_fb)[caf_i];
    auto caf_b = forward ? std::get<1>(caf_fb)[caf_i] : std::get<0>(caf_fb)[caf_i];

    bool only_max = false;

    const Joint& start_j = ann[start_i];
    Joint new_j = grow_connection_blend(
        caf_f, start_j.x, start_j.y, start_j.s, only_max);
    if (new_j.v == 0.0) return { 0.0, 0.0, 0.0, 0.0 };

    new_j.v = sqrt(new_j.v * start_j.v);  // geometric mean
    if (new_j.v < keypoint_threshold)
        return { 0.0, 0.0, 0.0, 0.0 };
    if (new_j.v < start_j.v * keypoint_threshold_rel)
        return { 0.0, 0.0, 0.0, 0.0 };

    // reverse match
    if (reverse_match && reverse_match_) {
        Joint reverse_j = grow_connection_blend(
            caf_b, new_j.x, new_j.y, new_j.s, only_max);
        if (reverse_j.v == 0.0)
            return { 0.0, 0.0, 0.0, 0.0 };
        if (fabs(start_j.x - reverse_j.x) + fabs(start_j.y - reverse_j.y) > start_j.s)
            return { 0.0, 0.0, 0.0, 0.0 };
    }

    return new_j;
}


void CifCaf::_force_complete(
    std::vector<std::vector<Joint> >* annotations,
    const torch::Tensor& cifhr_accumulated, double cifhr_revision,
    const torch::Tensor& caf_field, int64_t caf_stride
) {
    utils::CafScored caf_scored(cifhr_accumulated, cifhr_revision, force_complete_caf_th, 0.1);
    caf_scored.fill(caf_field, caf_stride, skeleton);
    auto caf_fb = caf_scored.get();

    for (auto&& ann : *annotations) {
        _grow(&ann, caf_fb, false);
    }
}


void CifCaf::_flood_fill(std::vector<Joint>* ann) {
    while (!frontier.empty()) frontier.pop();

    // initialize frontier
    for (int64_t j=0; j < n_keypoints; j++) {
        if ((*ann)[j].v == 0.0) continue;
        _frontier_add_from(*ann, j);
    }

    while (!frontier.empty()) {
        FrontierEntry entry(frontier.top());
        frontier.pop();
        // Was the target already filled by something else?
        if ((*ann)[entry.end_i].v > 0.0) continue;

        (*ann)[entry.end_i] = (*ann)[entry.start_i];
        (*ann)[entry.end_i].v = 0.00001;
        _frontier_add_from(*ann, entry.end_i);
    }
}


}  // namespace decoder
}  // namespace openpifpaf
