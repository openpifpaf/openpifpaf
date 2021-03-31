#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct CifHr : torch::CustomClassHolder {
    torch::Tensor accumulated;
    static int64_t neighbors;
    static double v_threshold;

    CifHr(
        const at::IntArrayRef& shape,
        int64_t stride
    ) : accumulated(torch::zeros({
            shape[0],
            (shape[1] - 1) * stride + 1,
            (shape[2] - 1) * stride + 1,
        }))
    { }

    void accumulate(const torch::Tensor& cif_field, int64_t stride, double min_scale=0.0, double factor=1.0);
    void add_gauss(int64_t f, double v, double x, double y, double sigma, double truncate=1.0);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
