#pragma once

#include <torch/script.h>

#include <memory>
#include <vector>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"
#include "openpifpaf/decoder/cifdet.hpp"  // for Detection type


namespace openpifpaf {
namespace decoder {
namespace utils {


struct NMSDetection : torch::CustomClassHolder {
    static double suppression;
    static double suppression_soft;
    static double instance_threshold;
    static double iou_threshold;
    static double iou_threshold_soft;

    STATIC_GETSET(double, suppression)
    STATIC_GETSET(double, suppression_soft)
    STATIC_GETSET(double, instance_threshold)
    STATIC_GETSET(double, iou_threshold)
    STATIC_GETSET(double, iou_threshold_soft)

    NMSDetection() { }

    void call(Occupancy* occupancy, std::vector<Detection>* annotations);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
