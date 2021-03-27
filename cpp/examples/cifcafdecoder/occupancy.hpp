#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct Occupancy : torch::CustomClassHolder {
    double reduction;
    double min_scale_reduced;

    Occupancy(
        double reduction,
        double min_scale
    ) : reduction(reduction),
        min_scale_reduced(min_scale / reduction)
    { }

    torch::Tensor forward(const torch::Tensor& x);
    void set(torch::Tensor& o, double x, double y, double sigma);
    unsigned char get(const torch::Tensor& o, double x, double y);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
