#include <algorithm>
#include <cmath>
#include <queue>

#include "openpifpaf/decoder/cifcaf.hpp"

#include "openpifpaf/decoder/utils/caf_scored.hpp"
#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/cif_seeds.hpp"
#include "openpifpaf/decoder/utils/nms_keypoints.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {


bool CifCaf::block_joints = false;
bool CifCaf::greedy = false;
double CifCaf::keypoint_threshold = 0.15;
double CifCaf::keypoint_threshold_rel = 0.5;
bool CifCaf::reverse_match = true;
bool CifCaf::force_complete = false;
double CifCaf::force_complete_caf_th = 0.001;


bool FrontierCompare::operator() (const FrontierEntry& a, const FrontierEntry& b) {
    return a.max_score < b.max_score;
}


Joint grow_connection_blend(const torch::Tensor& caf,
                            double x,
                            double y,
                            double xy_scale,
                            double filter_sigmas,
                            bool only_max) {
    /*
    Blending the top two candidates with a weighted average.

    Similar to the post processing step in
    "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs".
    */
    xy_scale = fmax(xy_scale, 0.5);  // enforce a minimum scale

    auto caf_a = caf.accessor<float, 2>();
    float sigma_filter = filter_sigmas * xy_scale / 2.0;
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

    float entry_1[3] = {  // xys
        caf_a[score_1_i][3], caf_a[score_1_i][4],
        fmaxf(0.0f, caf_a[score_1_i][6])
    };
    if (only_max)
        return { score_1, entry_1[0], entry_1[1], entry_1[2] };
    if (score_2 < 0.01 || score_2 < 0.5 * score_1)
        return { 0.5 * score_1, entry_1[0], entry_1[1], entry_1[2] };

    // blend
    float entry_2[3] = {  // xys
        caf_a[score_2_i][3], caf_a[score_2_i][4],
        fmaxf(0.0f, caf_a[score_2_i][6])
    };

    float blend_d2 = std::pow(entry_1[0] - entry_2[0], 2) + std::pow(entry_1[1] - entry_2[1], 2);
    if (blend_d2 > std::pow(entry_1[2], 2) / 4.0)
        return { 0.5 * score_1, entry_1[0], entry_1[1], entry_1[2] };

    return {
        0.5 * (score_1 + score_2),
        (score_1 * entry_1[0] + score_2 * entry_2[0]) / (score_1 + score_2),
        (score_1 * entry_1[1] + score_2 * entry_2[1]) / (score_1 + score_2),
        (score_1 * entry_1[2] + score_2 * entry_2[2]) / (score_1 + score_2)
    };
}

std::vector<double> grow_connection_blend_py(const torch::Tensor& caf,
                                             double x,
                                             double y,
                                             double s,
                                             double filter_sigmas,
                                             bool only_max) {
    Joint j(grow_connection_blend(caf, x, y, s, filter_sigmas, only_max));
    return { j.x, j.y, j.s, j.v };  // xysv
}


std::tuple<torch::Tensor, torch::Tensor> CifCaf::call(
    const torch::Tensor& cif_field,
    int64_t cif_stride,
    const torch::Tensor& caf_field,
    int64_t caf_stride
) {
    return call_with_initial_annotations(cif_field, cif_stride, caf_field, caf_stride, torch::nullopt, torch::nullopt);
}


std::tuple<torch::Tensor, torch::Tensor> CifCaf::call_with_initial_annotations(
    const torch::Tensor& cif_field,
    int64_t cif_stride,
    const torch::Tensor& caf_field,
    int64_t caf_stride,
    torch::optional<torch::Tensor> initial_annotations,
    torch::optional<torch::Tensor> initial_ids
) {
#ifdef DEBUG
    TORCH_WARN("cpp CifCaf::call()");
#endif
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
#ifdef DEBUG
    TORCH_WARN("seeds=", seeds_f_a.size(0));
#endif

    utils::CafScored caf_scored(cifhr_accumulated, cifhr_revision, -1.0, 0.1);
#ifdef DEBUG
    TORCH_WARN("caf scored created");
#endif
    caf_scored.fill(caf_field, caf_stride, skeleton);
#ifdef DEBUG
    TORCH_WARN("caf scored filled");
#endif
    auto caf_fb = caf_scored.get();
#ifdef DEBUG
    TORCH_WARN("caf scored done");
    auto caf_f = std::get<0>(caf_fb);
    int64_t n_caf_f = std::accumulate(caf_f.begin(), caf_f.end(), 0,
                                      [](int64_t a, torch::Tensor& b) { return a + b.size(0); });
    auto caf_b = std::get<1>(caf_fb);
    int64_t n_caf_b = std::accumulate(caf_b.begin(), caf_b.end(), 0,
                                      [](int64_t a, torch::Tensor& b) { return a + b.size(0); });
    TORCH_WARN("caf forward=", n_caf_f, " caf backward=", n_caf_b);
#endif

    occupancy.reset(cifhr_accumulated.sizes());
    std::vector<Annotation> annotations;

    // process initial annotations first
    if (initial_annotations.has_value()) {
        TORCH_CHECK(initial_ids.has_value(), "require initial_ids when initial_annotations are given");

        auto initial_annotations_a = initial_annotations->accessor<float, 3>();
        auto initial_ids_a = initial_ids->accessor<int64_t, 1>();
        for (int64_t ann_i=0; ann_i < initial_annotations_a.size(0); ann_i++) {
            Annotation annotation(n_keypoints, initial_ids_a[ann_i]);
            auto ann = initial_annotations_a[ann_i];
            for (int64_t of=0; of < n_keypoints; of++) {
                Joint& o_joint = annotation.joints[of];
                o_joint.v = ann[of][0];
                o_joint.x = ann[of][1];
                o_joint.y = ann[of][2];
                o_joint.s = ann[of][3];
            }

            _grow(&annotation, caf_fb);

            for (int64_t of=0; of < occupancy.n_fields(); of++) {
                Joint& o_joint = annotation.joints[of];
                if (o_joint.v == 0.0) continue;
                occupancy.set(of, o_joint.x, o_joint.y, o_joint.s);
            }
            annotations.push_back(annotation);
        }
    }

    int64_t f;
    float x, y, s;
    for (int64_t seed_i=0; seed_i < seeds_f.size(0); seed_i++) {
        f = seeds_f_a[seed_i];
        x = seeds_vxys_a[seed_i][1];
        y = seeds_vxys_a[seed_i][2];
        s = seeds_vxys_a[seed_i][3];
        if (occupancy.get(f, x, y)) continue;

        Annotation annotation(n_keypoints);
        Joint& joint = annotation.joints[f];
        joint.v = seeds_vxys_a[seed_i][0];
        joint.x = x;
        joint.y = y;
        joint.s = s;
#ifdef DEBUG
        TORCH_WARN("new seed: field=", f, " x=", x, " y=", y, " s=", s);
#endif

        _grow(&annotation, caf_fb);

        for (int64_t of=0; of < occupancy.n_fields(); of++) {
            Joint& o_joint = annotation.joints[of];
            if (o_joint.v == 0.0) continue;
            occupancy.set(of, o_joint.x, o_joint.y, o_joint.s);
        }
        annotations.push_back(annotation);
    }

    if (force_complete) {
        _force_complete(&annotations, cifhr_accumulated, cifhr_revision, caf_field, caf_stride);
        for (auto& ann : annotations) _flood_fill(&ann);
    }

#ifdef DEBUG
    TORCH_WARN("NMS");
#endif
    utils::NMSKeypoints().call(&occupancy, &annotations);

#ifdef DEBUG
    TORCH_WARN("convert to tensor");
#endif
    auto out = torch::empty({ int64_t(annotations.size()), n_keypoints, 4 });
    auto out_ids = torch::empty({ int64_t(annotations.size()) }, torch::kInt64);
    auto out_a = out.accessor<float, 3>();
    auto out_ids_a = out_ids.accessor<int64_t, 1>();
    for (int64_t ann_i = 0; ann_i < int64_t(annotations.size()); ann_i++) {
        auto& ann = annotations[ann_i];
        for (int64_t joint_i = 0; joint_i < n_keypoints; joint_i++) {
            Joint& joint = ann.joints[joint_i];
            out_a[ann_i][joint_i][0] = joint.v;
            out_a[ann_i][joint_i][1] = joint.x;
            out_a[ann_i][joint_i][2] = joint.y;
            out_a[ann_i][joint_i][3] = joint.s;
        }
        out_ids_a[ann_i] = ann.id;
    }
    return std::make_tuple(out, out_ids);
}


void CifCaf::_grow(
    Annotation* ann,
    const caf_fb_t& caf_fb,
    bool reverse_match_,
    double filter_sigmas
) {
    while (!frontier.empty()) frontier.pop();
    in_frontier.clear();

    // initialize frontier
    for (int64_t j=0; j < n_keypoints; j++) {
        if (ann->joints[j].v == 0.0) continue;
        _frontier_add_from(*ann, j);
    }

    while (!frontier.empty()) {
        FrontierEntry entry(frontier.top());
        frontier.pop();
        // Was the target already filled by something else?
        if (ann->joints[entry.end_i].v > 0.0) continue;

        // Is entry not fully computed?
        if (entry.joint.v == 0.0) {
            Joint new_joint = _connection_value(
                *ann, caf_fb, entry.start_i, entry.end_i, reverse_match_, filter_sigmas);
            if (new_joint.v == 0.0) {
                if (block_joints) {
                    new_joint.v = 0.00001;
                    entry.joint = new_joint;
                }
                continue;
            }

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

        ann->joints[entry.end_i] = entry.joint;
        _frontier_add_from(*ann, entry.end_i);
    }
}


void CifCaf::_frontier_add_from(
    const Annotation& ann,
    int64_t start_i
) {
    float max_score = sqrt(ann.joints[start_i].v);

    auto skeleton_a = skeleton.accessor<int64_t, 2>();
    for (int64_t f=0; f < skeleton_a.size(0); f++) {
        int64_t pair_0 = skeleton_a[f][0];
        int64_t pair_1 = skeleton_a[f][1];

        if (pair_0 == start_i) {
            if (ann.joints[pair_1].v > 0.0) continue;
            if (in_frontier.find(std::make_pair(pair_0, pair_1)) != in_frontier.end()) {
                continue;
            }
            frontier.emplace(max_score, pair_0, pair_1);
            in_frontier.emplace(pair_0, pair_1);
            continue;
        }
        if (pair_1 == start_i) {
            if (ann.joints[pair_0].v > 0.0) continue;
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
    const Annotation& ann,
    const caf_fb_t& caf_fb,
    int64_t start_i,
    int64_t end_i,
    bool reverse_match_,
    double filter_sigmas
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

    bool only_max = false;  // TODO make configurable

    const Joint& start_j = ann.joints[start_i];
    Joint new_j = grow_connection_blend(
        caf_f, start_j.x, start_j.y, start_j.s, filter_sigmas, only_max);
    if (new_j.v == 0.0) return new_j;

    new_j.v = sqrt(new_j.v * start_j.v);  // geometric mean
    if (new_j.v < keypoint_threshold || new_j.v < start_j.v * keypoint_threshold_rel) {
        new_j.v = 0.0;
        return new_j;
    }

    // reverse match
    // Only compute when requested and when the start joint is within the
    // occupancy and HR maps. For forward tracking, the source joints
    // are not part of the predicted fields and therefore reverse matching
    // cannot work in these cases.
    if (reverse_match && reverse_match_ && start_i < occupancy.n_fields()) {
        Joint reverse_j = grow_connection_blend(
            caf_b, new_j.x, new_j.y, new_j.s, filter_sigmas, only_max);
        if (reverse_j.v == 0.0) {
            new_j.v = 0.0;
            return new_j;
        }
        if (fabs(start_j.x - reverse_j.x) + fabs(start_j.y - reverse_j.y) > start_j.s) {
            new_j.v = 0.0;
            return new_j;
        }
    }

    return new_j;
}


void CifCaf::_force_complete(
    std::vector<Annotation>* annotations,
    const torch::Tensor& cifhr_accumulated, double cifhr_revision,
    const torch::Tensor& caf_field, int64_t caf_stride
) {
    utils::CafScored caf_scored(cifhr_accumulated, cifhr_revision, force_complete_caf_th, 0.1);
    caf_scored.fill(caf_field, caf_stride, skeleton);
    auto caf_fb = caf_scored.get();

    for (auto& ann : *annotations) {
        _grow(&ann, caf_fb, false, 4.0);
    }
}


void CifCaf::_flood_fill(Annotation* ann) {
    while (!frontier.empty()) frontier.pop();
    in_frontier.clear();

    // initialize frontier
    for (int64_t j=0; j < n_keypoints; j++) {
        if (ann->joints[j].v == 0.0) continue;
        _frontier_add_from(*ann, j);
    }

    while (!frontier.empty()) {
        FrontierEntry entry(frontier.top());
        frontier.pop();
        // Was the target already filled by something else?
        if (ann->joints[entry.end_i].v > 0.0) continue;

        ann->joints[entry.end_i] = ann->joints[entry.start_i];
        ann->joints[entry.end_i].v = 0.00001;
        _frontier_add_from(*ann, entry.end_i);
    }
}


}  // namespace decoder
}  // namespace openpifpaf
