#include <algorithm>
#include <cmath>

#include "openpifpaf/decoder/utils/nms_keypoints.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSKeypoints::suppression = 0.00001;  // larger zero for force-complete
double NMSKeypoints::instance_threshold = 0.15;
double NMSKeypoints::keypoint_threshold = 0.15;


void NMSKeypoints::call(Occupancy* occupancy, std::vector<Annotation>* annotations) {
    occupancy->clear();

    std::sort(
        annotations->begin(),
        annotations->end(),
        [&](const Annotation& a, const Annotation& b) {
            return (score->value(a) > score->value(b));
        }
    );

    int64_t n_occupancy = occupancy->occupancy.size(0);
    for (auto& ann : *annotations) {
        TORCH_CHECK(n_occupancy <= int64_t(ann.joints.size()),
                    "NMS occupancy map must be of same size or smaller as annotation");

        int64_t f = -1;
        for (Joint& joint : ann.joints) {
            f++;
            if (f >= n_occupancy) break;
            if (joint.v == 0.0) continue;
            if (occupancy->get(f, joint.x, joint.y)) {
                joint.v *= suppression;
            } else {
                occupancy->set(f, joint.x, joint.y, joint.s);  // joint.s = 2 * sigma
            }
        }
    }

    // suppress below keypoint threshold
    for (auto& ann : *annotations) {
        for (Joint& joint : ann.joints) {
            if (joint.v > keypoint_threshold) continue;
            joint.v = 0.0;
        }
    }

    // remove annotations below instance threshold
    annotations->erase(
        std::remove_if(annotations->begin(), annotations->end(), [&](const Annotation& ann) {
            return (score->value(ann) < instance_threshold);
        }),
        annotations->end()
    );

    std::sort(
        annotations->begin(),
        annotations->end(),
        [&](const Annotation& a, const Annotation& b) {
            return (score->value(a) > score->value(b));
        }
    );
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
