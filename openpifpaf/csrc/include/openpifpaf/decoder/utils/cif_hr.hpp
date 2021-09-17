#pragma once

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "openpifpaf/utils.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


struct CifHr : torch::CustomClassHolder {
    at::Tensor accumulated_buffer;
    at::Tensor accumulated;
    double revision;
    static int64_t neighbors;
    static double threshold;
    static bool ablation_skip;

    CifHr(
    ) : accumulated_buffer(at::zeros({ 1, 1, 1 })),
        accumulated(
            accumulated_buffer.index({
                at::indexing::Slice(0, 1),
                at::indexing::Slice(0, 1),
                at::indexing::Slice(0, 1)
            })
        ),
        revision(0.0)
    { }

    void accumulate(const at::Tensor& cif_field, int64_t stride, double min_scale = 0.0, double factor = 1.0);
    void add_gauss(int64_t f, float v, float x, float y, float sigma, float truncate = 1.0);
    std::tuple<at::Tensor, double> get_accumulated(void);
    void reset(const at::IntArrayRef& shape, int64_t stride);
};


struct CifDetHr : CifHr {
    void accumulate(const at::Tensor& cifdet_field, int64_t stride, double min_scale = 0.0, double factor = 1.0);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
