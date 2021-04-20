#include <algorithm>
#include <math.h>

#include "openpifpaf/decoder/utils/nms.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSKeypoints::suppression = 0.0;
double NMSKeypoints::instance_threshold = 0.15;
double NMSKeypoints::keypoint_threshold = 0.15;


struct AnnotationCompare {
    bool operator() (const std::vector<Joint>& a, const std::vector<Joint>& b) {
        float score_a = std::accumulate(a.begin(), a.end(), 0.0f, [](float i, const Joint& j) { return i + j.v; }) / a.size();
        float score_b = std::accumulate(b.begin(), b.end(), 0.0f, [](float i, const Joint& j) { return i + j.v; }) / a.size();
        return (score_a < score_b);
    }
} annotation_compare;


void NMSKeypoints::call(Occupancy& occupancy, std::vector<std::vector<Joint> >& annotations) {
    occupancy.clear();
    std::sort(annotations.begin(), annotations.end(), annotation_compare);

    for (auto&& ann : annotations) {
        TORCH_CHECK(occupancy.occupancy.size(0) == ann.size(), "NMS occupancy map must be of same size as annotation");

        int64_t f = -1;
        for (Joint& joint : ann) {
            f++;
            if (joint.v == 0.0) continue;
            if (occupancy.get(f, joint.x, joint.y)) {
                joint.v *= suppression;
            } else {
                occupancy.set(f, joint.x, joint.y, joint.s);  // joint.s = 2 * sigma
            }
        }
    }

    // suppress below keypoint threshold
    for (auto&& ann : annotations) {
        for (Joint& joint : ann) {
            if (joint.v > keypoint_threshold) continue;
            joint.v = 0.0;
        }
    }

    // remove annotations below instance threshold
    annotations.erase(
        std::remove_if(annotations.begin(), annotations.end(), [](const std::vector<Joint>& ann) {
            float score = std::accumulate(ann.begin(), ann.end(), 0.0f, [](float i, const Joint& j) { return i + j.v; }) / ann.size();
            return (score < instance_threshold);
        }),
        annotations.end()
    );

    std::sort(annotations.begin(), annotations.end(), annotation_compare);
}


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
