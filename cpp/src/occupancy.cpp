#include <algorithm>
#include <math.h>

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
    occupancy.index_put_({f, at::indexing::Slice(miny, maxy), at::indexing::Slice(minx, maxx)}, 1);
}


bool Occupancy::get(int64_t f, double x, double y) {
    if (f >= occupancy.size(0)) return 1;

    if (reduction != 1.0) {
        x /= reduction;
        y /= reduction;
    }

    auto xi = std::clamp(int64_t(x), int64_t(0), occupancy.size(1) - 1);
    auto yi = std::clamp(int64_t(y), int64_t(0), occupancy.size(0) - 1);
    return occupancy.index({yi, xi}).item<uint8_t>() != 0;
}


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
