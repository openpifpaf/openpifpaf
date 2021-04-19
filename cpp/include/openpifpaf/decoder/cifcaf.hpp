#pragma once

#include <algorithm>
#include <queue>

#include <torch/script.h>

#include "openpifpaf/decoder/utils/cif_hr.hpp"


namespace openpifpaf {
namespace decoder {


struct Joint {
    double v, x, y, s;

    Joint(void) : v(0.0), x(0.0), y(0.0), s(0.0) { }
    Joint(double v_, double x_, double y_, double s_)
    : v(v_), x(x_), y(y_), s(s_)
    { }
};


std::vector<double> grow_connection_blend_py(const torch::Tensor& caf, double x, double y, double s, bool only_max);


struct FrontierEntry {
    float max_score;
    Joint joint;
    int64_t start_i;
    int64_t end_i;

    FrontierEntry(float max_score_, int64_t start_i_, int64_t end_i_)
    : max_score(max_score_), start_i(start_i_), end_i(end_i_) { }
    FrontierEntry(float max_score_, Joint joint_, int64_t start_i_, int64_t end_i_)
    : max_score(max_score_), joint(joint_), start_i(start_i_), end_i(end_i_) { }
};
auto frontier_compare = [](FrontierEntry& a, FrontierEntry& b) { return a.max_score < b.max_score; };


struct IntPairHash
{
    std::size_t operator()(std::pair<int64_t, int64_t> const& p) const noexcept
    {
        std::size_t h1 = std::hash<int64_t>{}(p.first);
        std::size_t h2 = std::hash<int64_t>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};


typedef std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor> > caf_fb_t;


struct CifCaf : torch::CustomClassHolder {
    int64_t n_keypoints;
    std::vector<std::vector<int64_t> > skeleton;
    static bool greedy;
    static double keypoint_threshold;
    static double keypoint_threshold_rel;
    static bool global_reverse_match;

    utils::CifHr cifhr;
    std::priority_queue<FrontierEntry, std::vector<FrontierEntry>, decltype(frontier_compare)> frontier;
    std::unordered_set<std::pair<int64_t, int64_t>, IntPairHash > in_frontier;

    CifCaf(
        int64_t n_keypoints_,
        const std::vector<std::vector<int64_t> >& skeleton_
    ) :
        n_keypoints(n_keypoints_),
        skeleton(skeleton_),
        cifhr({ 1, 1, 1, 1 }, 1),
        frontier(frontier_compare)
    { }

    torch::Tensor call(
        const torch::Tensor& cif_field,
        int64_t cif_stride,
        const torch::Tensor& caf_field,
        int64_t caf_stride
    );

    void _grow(std::vector<Joint>& ann, const caf_fb_t& caf_fb, bool reverse_match=true);
    void _frontier_add_from(std::vector<Joint>& ann, int64_t start_i);
    // FrontierEntry _frontier_get(std::vector<Joint>& ann, const caf_fb_t& caf_fb, bool reverse_match);

    Joint _connection_value(
        const std::vector<Joint>& ann,
        const caf_fb_t& caf_fb,
        int64_t start_i,
        int64_t end_i,
        bool reverse_match=true
    );
};


} // namespace decoder
} // namespace openpifpaf
