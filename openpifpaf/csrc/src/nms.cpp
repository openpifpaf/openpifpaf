#include <math.h>

#include <algorithm>

#include "openpifpaf/decoder/utils/nms.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSKeypoints::suppression = 0.0;
double NMSKeypoints::instance_threshold = 0.15;
double NMSKeypoints::keypoint_threshold = 0.15;


struct AnnotationCompare {
    std::shared_ptr<AnnotationScore> score;
    explicit AnnotationCompare(std::shared_ptr<AnnotationScore> score_) : score(score_) { }

    bool operator() (const std::vector<Joint>& a, const std::vector<Joint>& b) {
        return (score->value(a) > score->value(b));
    }
};


void NMSKeypoints::call(Occupancy* occupancy, std::vector<std::vector<Joint> >* annotations) {
    occupancy->clear();
    std::sort(annotations->begin(), annotations->end(), AnnotationCompare(score));

    for (auto&& ann : *annotations) {
        TORCH_CHECK(occupancy->occupancy.size(0) == int64_t(ann.size()),
                    "NMS occupancy map must be of same size as annotation");

        int64_t f = -1;
        for (Joint& joint : ann) {
            f++;
            if (joint.v == 0.0) continue;
            if (occupancy->get(f, joint.x, joint.y)) {
                joint.v *= suppression;
            } else {
                occupancy->set(f, joint.x, joint.y, joint.s);  // joint.s = 2 * sigma
            }
        }
    }

    // suppress below keypoint threshold
    for (auto&& ann : *annotations) {
        for (Joint& joint : ann) {
            if (joint.v > keypoint_threshold) continue;
            joint.v = 0.0;
        }
    }

    // remove annotations below instance threshold
    annotations->erase(
        std::remove_if(annotations->begin(), annotations->end(), [=](const std::vector<Joint>& ann) {
            return (score->value(ann) < instance_threshold);
        }),
        annotations->end()
    );

    std::sort(annotations->begin(), annotations->end(), AnnotationCompare(score));
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
