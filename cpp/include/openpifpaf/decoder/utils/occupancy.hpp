#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct Occupancy : torch::CustomClassHolder {
    torch::Tensor occupancy;
    double reduction;
    double min_scale_reduced;

    Occupancy(
        const at::IntArrayRef& shape,
        double reduction,
        double min_scale
    ) : occupancy(torch::zeros({
            shape[0],
            static_cast<int64_t>(shape[1] / reduction) + 1,
            static_cast<int64_t>(shape[2] / reduction) + 1,
        }, torch::kUInt8)),
        reduction(reduction),
        min_scale_reduced(min_scale / reduction)
    { }

    void set(int64_t f, double x, double y, double sigma);
    bool get(int64_t f, double x, double y);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
