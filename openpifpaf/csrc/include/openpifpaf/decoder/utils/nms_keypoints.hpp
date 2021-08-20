#pragma once

#include <torch/script.h>

#include <memory>
#include <vector>

#include "openpifpaf/utils.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"
#include "openpifpaf/decoder/cifcaf.hpp"  // for Joint and Annotation type


namespace openpifpaf {
namespace decoder {
namespace utils {


struct AnnotationScore {
    virtual double value(const Annotation& annotation) const = 0;
    virtual ~AnnotationScore() = default;
};


struct UniformScore : public AnnotationScore {
    double value(const Annotation& annotation) const {
        return std::accumulate(
            annotation.joints.begin(), annotation.joints.end(), 0.0,
            [](float i, const Joint& j) { return i + j.v; }
        ) / annotation.joints.size();
    }
};


struct NMSKeypoints : torch::CustomClassHolder {
    std::unique_ptr<AnnotationScore> score;

    static double suppression;
    static double instance_threshold;
    static double keypoint_threshold;

    NMSKeypoints() : score(std::make_unique<UniformScore>()) { }

    void call(Occupancy* occupancy, std::vector<Annotation>* annotations);
};


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
