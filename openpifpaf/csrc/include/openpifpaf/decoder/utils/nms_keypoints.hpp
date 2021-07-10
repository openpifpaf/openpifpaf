#pragma once

#include <torch/script.h>

#include <memory>
#include <vector>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"
#include "openpifpaf/decoder/cifcaf.hpp"  // for Joint type


namespace openpifpaf {
namespace decoder {
namespace utils {


struct AnnotationScore {
    virtual double value(const std::vector<Joint>& annotation) const = 0;
    virtual ~AnnotationScore() = default;
};


struct UniformScore : public AnnotationScore {
    double value(const std::vector<Joint>& annotation) const {
        std::cout << "-- score: ";
        for (auto& j : annotation) std::cout << j.v << ", ";
        double v = std::accumulate(
            annotation.begin(), annotation.end(), 0.0,
            [](float i, const Joint& j) { return i + j.v; }
        ) / annotation.size();
        std::cout << "score done: " << v << std::endl;
        return v;
    }
};


struct NMSKeypoints : torch::CustomClassHolder {
    AnnotationScore* score;

    static double suppression;
    static double instance_threshold;
    static double keypoint_threshold;

    STATIC_GETSET(double, suppression)
    STATIC_GETSET(double, instance_threshold)
    STATIC_GETSET(double, keypoint_threshold)

    NMSKeypoints() : score(new UniformScore()) { }

    void call(Occupancy* occupancy, std::vector<std::vector<Joint> >* annotations);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
