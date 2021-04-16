#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {


std::vector<double> grow_connection_blend(const torch::Tensor& caf, double x, double y, double s, bool only_max);


} // namespace decoder
} // namespace openpifpaf
