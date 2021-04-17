#pragma once

#include "openpifpaf/decoder/utils/cif_hr.hpp"

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {


struct Joint {
    double v, x, y, s;

    Joint(void) : v(0.0), x(0.0), y(0.0), s(0.0) { }
    Joint(double v_, double x_, double y_, double s_)
    : v(v_), x(x_), y(y_), s(s_)
    { }
};


struct Annotation : torch::CustomClassHolder {
    const std::vector<std::string>& keypoints;
    const std::vector<std::vector<int64_t> >& skeleton;
    std::vector<Joint> joints;

    Annotation(
        const std::vector<std::string>& keypoints_,
        const std::vector<std::vector<int64_t> >& skeleton_
    ) :
        keypoints(keypoints_),
        skeleton(skeleton_),
        joints(keypoints_.size())
    { }
};


std::vector<double> grow_connection_blend(const torch::Tensor& caf, double x, double y, double s, bool only_max);


struct CifCaf : torch::CustomClassHolder {
    std::vector<std::string> keypoints;
    std::vector<std::vector<int64_t> > skeleton;
    std::vector<std::vector<int64_t> > out_skeleton;

    utils::CifHr cifhr;

    CifCaf(
        const std::vector<std::string>& keypoints_,
        const std::vector<std::vector<int64_t> >& skeleton_,
        const std::vector<std::vector<int64_t> >& out_skeleton_
    ) :
        keypoints(keypoints_),
        skeleton(skeleton_),
        out_skeleton(out_skeleton_),
        cifhr({ 1, 1, 1, 1 }, 1)
    { }

    std::vector<c10::intrusive_ptr<Annotation> > call(
        const torch::Tensor& cif_field,
        int64_t cif_stride,
        const torch::Tensor& caf_field,
        int64_t caf_stride
    );
};


} // namespace decoder
} // namespace openpifpaf
