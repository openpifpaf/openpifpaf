#pragma once

#include <torch/script.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include "openpifpaf/utils.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


void cif_hr_add_gauss_op(const torch::Tensor& accumulated,
                         float accumulated_revision,
                         int64_t f,
                         float v,
                         float x,
                         float y,
                         float sigma,
                         float truncate = 3.0);


struct CifHr : torch::CustomClassHolder {
    torch::Tensor accumulated_buffer;
    torch::Tensor accumulated;
    double revision;
    static int64_t neighbors;
    static double threshold;
    static bool ablation_skip;

    CifHr(
    ) : accumulated_buffer(torch::zeros({ 1, 1, 1 })),
        accumulated(
            accumulated_buffer.index({
                at::indexing::Slice(0, 1),
                at::indexing::Slice(0, 1),
                at::indexing::Slice(0, 1)
            })
        ),
        revision(0.0)
    { }

    void accumulate(const torch::Tensor& cif_field, int64_t stride, double min_scale = 0.0, double factor = 1.0);
    void add_gauss(int64_t f, float v, float x, float y, float sigma, float truncate = 1.0);
    std::tuple<torch::Tensor, double> get_accumulated(void);
    void reset(const at::IntArrayRef& shape, int64_t stride);
};


struct CifDetHr : CifHr {
    void accumulate(const torch::Tensor& cifdet_field, int64_t stride, double min_scale = 0.0, double factor = 1.0);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
