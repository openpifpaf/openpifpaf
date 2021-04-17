#pragma once

#include <algorithm>
#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


void cif_hr_accumulate_op(const torch::Tensor& accumulated,
                          double accumulated_revision,
                          const torch::Tensor& cif_field,
                          int64_t stride,
                          double v_threshold,
                          int64_t neighbors,
                          double min_scale=0.0,
                          double factor=1.0);


void cif_hr_add_gauss_op(const torch::Tensor& accumulated,
                         float accumulated_revision,
                         int64_t f,
                         float v,
                         float x,
                         float y,
                         float sigma,
                         float truncate=3.0);


struct CifHr : torch::CustomClassHolder {
    torch::Tensor accumulated_buffer;
    torch::Tensor accumulated;
    double revision;
    static int64_t neighbors;
    static double v_threshold;

    CifHr(
        const at::IntArrayRef& shape,
        int64_t stride
    ) : accumulated_buffer(torch::zeros({
            shape[0],
            (std::max(shape[2], shape[3]) - 1) * stride + 1,
            (std::max(shape[2], shape[3]) - 1) * stride + 1
        })),
        accumulated(
            accumulated_buffer.index({
                at::indexing::Slice(0, shape[0]),
                at::indexing::Slice(0, (shape[2] - 1) * stride + 1),
                at::indexing::Slice(0, (shape[3] - 1) * stride + 1)
            })
        ),
        revision(0.0)
    { }

    void accumulate(const torch::Tensor& cif_field, int64_t stride, double min_scale=0.0, double factor=1.0);
    void add_gauss(int64_t f, double v, double x, double y, double sigma, double truncate=1.0);
    std::tuple<torch::Tensor, double> get_accumulated(void);
    void reset(const at::IntArrayRef& shape, int64_t stride);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
