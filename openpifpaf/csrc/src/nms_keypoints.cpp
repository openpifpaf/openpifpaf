#include <math.h>

#include <algorithm>

#include "openpifpaf/decoder/utils/nms_keypoints.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSKeypoints::suppression = 0.0;
double NMSKeypoints::instance_threshold = 0.15;
double NMSKeypoints::keypoint_threshold = 0.15;


struct AnnotationCompare {
    const std::shared_ptr<AnnotationScore>& score;

    explicit AnnotationCompare(const std::shared_ptr<AnnotationScore>& score_) : score(score_) { }

    bool operator() (const std::vector<Joint>& a, const std::vector<Joint>& b) const {
        // return a.size() < b.size();
        // std::cout << "op" << score << std::endl;
        return (score->value(a) > score->value(b));
    }
};


void NMSKeypoints::call(Occupancy* occupancy, std::vector<std::vector<Joint> >* annotations) {
    occupancy->clear();
    std::cout << "test sort" << std::endl;
    std::vector<int64_t> test_vector{ 8, 7, 1, 2, 3, 4 };
    std::sort(test_vector.begin(), test_vector.end());
    std::cout << test_vector[0] << std::endl;
    std::cout << "score: " << score->value((*annotations)[0]) << std::endl;
    std::cout << "score done" << std::endl;
    std::cout << AnnotationCompare(score)((*annotations)[0], (*annotations)[0]) << std::endl;
    auto ann_compare = AnnotationCompare(score);
    std::cout << "anns" << std::endl;
    for (auto&& ann : *annotations) {
        std::cout << ann[0].v << ", " << ann[1].v << ", " << ann[2].v << ", " << std::endl;
    }
    std::cout << "before sort" << std::endl;
    std::sort(annotations->begin(), annotations->end(), ann_compare);
    // std::sort(
    //     annotations->begin(),
    //     annotations->end(),
    //     [](const std::vector<Joint> & a, const std::vector<Joint> & b) {
    //         return a.size() < b.size();
    //     }
    // );

    std::cout << "sort done" << std::endl;
    TORCH_WARN("nms 1: ", annotations->size());

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

    TORCH_WARN("nms 2: ", annotations->size());
    // suppress below keypoint threshold
    for (auto&& ann : *annotations) {
        for (Joint& joint : ann) {
            if (joint.v > keypoint_threshold) continue;
            joint.v = 0.0;
        }
    }

    TORCH_WARN("nms 3: ", annotations->size());
    // remove annotations below instance threshold
    annotations->erase(
        std::remove_if(annotations->begin(), annotations->end(), [=](const std::vector<Joint>& ann) {
            double s = score->value(ann);
            std::cout << "--- " << s << " >> " << instance_threshold << std::endl;
            return s < instance_threshold;
        }),
        annotations->end()
    );

    TORCH_WARN("nms 4: ", annotations->size());
    std::sort(annotations->begin(), annotations->end(), ann_compare);
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
