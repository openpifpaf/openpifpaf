#include <algorithm>
#include <cmath>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


void Occupancy::set(int64_t f, double x, double y, double sigma) {
    if (reduction != 1.0) {
        x /= reduction;
        y /= reduction;
        sigma = fmax(min_scale_reduced, sigma / reduction);
    }

    auto minx = std::clamp(int64_t(x - sigma), int64_t(0), occupancy.size(2) - 1);
    auto miny = std::clamp(int64_t(y - sigma), int64_t(0), occupancy.size(1) - 1);
    // # +1: for non-inclusive boundary
    // # There is __not__ another plus one for rounding up:
    // # The query in occupancy does not round to nearest integer but only
    // # rounds down.
    auto maxx = std::clamp(int64_t(x + sigma), minx + 1, occupancy.size(2));
    auto maxy = std::clamp(int64_t(y + sigma), miny + 1, occupancy.size(1));
    occupancy.index_put_({f, at::indexing::Slice(miny, maxy), at::indexing::Slice(minx, maxx)}, revision + 1);
}


bool Occupancy::get(int64_t f, double x, double y) {
    if (f >= occupancy.size(0)) return 1;

    if (reduction != 1.0) {
        x /= reduction;
        y /= reduction;
    }

    auto xi = std::clamp(int64_t(x), int64_t(0), occupancy.size(2) - 1);
    auto yi = std::clamp(int64_t(y), int64_t(0), occupancy.size(1) - 1);
    return occupancy.index({f, yi, xi}).item<int16_t>() > revision;
}


void Occupancy::reset(const at::IntArrayRef& shape) {
    auto j = static_cast<int64_t>(shape[1] / reduction) + 1;
    auto i = static_cast<int64_t>(shape[2] / reduction) + 1;
    if (occupancy_buffer.size(0) < shape[0]
        || occupancy_buffer.size(1) < j
        || occupancy_buffer.size(2) < i
    ) {
        OPENPIFPAF_INFO("resizing occupancy buffer");
        occupancy_buffer = torch::zeros({
            shape[0],
            std::max(j, i),
            std::max(j, i)
        }, torch::kInt16);
    }

    occupancy = occupancy_buffer.index({
        at::indexing::Slice(0, shape[0]),
        at::indexing::Slice(0, j),
        at::indexing::Slice(0, i)
    });

    clear();
}


void Occupancy::clear(void) {
    revision++;
    if (revision > 32000) {
        occupancy_buffer.zero_();
        revision = 0;
    }
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
