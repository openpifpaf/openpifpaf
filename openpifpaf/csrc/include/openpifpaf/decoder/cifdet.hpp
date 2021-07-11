#pragma once

#include <torch/script.h>

#include <algorithm>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"

namespace openpifpaf {
namespace decoder {


struct Detection {
    int64_t c;
    double v, x, y, w, h;

    Detection(void) : c(0), v(0.0), x(0.0), y(0.0), w(0.0), h(0.0) { }
    Detection(int64_t c_, double v_, double x_, double y_, double w_, double h_)
    : c(c_), v(v_), x(x_), y(y_), w(w_), h(h_)
    { }
};


struct OPENPIFPAF_API CifDet : torch::CustomClassHolder {
    static int64_t max_detections_before_nms;

    utils::CifDetHr cifDetHr;
    utils::Occupancy occupancy;

    CifDet(void) : cifDetHr(), occupancy(2.0, 4.0) { }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> call(
        const torch::Tensor& cifdet_field,
        int64_t cifdet_stride
    );
};


}  // namespace decoder
}  // namespace openpifpaf
