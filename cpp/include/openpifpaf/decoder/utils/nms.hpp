#pragma once

#include <torch/script.h>

#include "openpifpaf/decoder/utils/occupancy.hpp"
#include "openpifpaf/decoder/cifcaf.hpp"  // for Joint type


namespace openpifpaf {
namespace decoder {
namespace utils {


struct NMSKeypoints : torch::CustomClassHolder {
    static double suppression;
    static double instance_threshold;
    static double keypoint_threshold;

    void call(Occupancy& occupancy, std::vector<std::vector<Joint> >& annotations);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
