#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct Occupancy : torch::CustomClassHolder {
    torch::Tensor occupancy_buffer;
    torch::Tensor occupancy;
    double reduction;
    double min_scale_reduced;
    int64_t revision;

    Occupancy(
        double reduction,
        double min_scale
    ) : occupancy_buffer(torch::zeros({ 1, 1, 1 }, torch::kInt16)),
        occupancy(occupancy_buffer.index({
            at::indexing::Slice(0, 1),
            at::indexing::Slice(0, 1),
            at::indexing::Slice(0, 1)
        })),
        reduction(reduction),
        min_scale_reduced(min_scale / reduction),
        revision(0)
    { }

    void set(int64_t f, double x, double y, double sigma);
    bool get(int64_t f, double x, double y);
    int64_t n_fields(void) { return occupancy.size(0); }
    void reset(const at::IntArrayRef& shape);
    void clear(void);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
