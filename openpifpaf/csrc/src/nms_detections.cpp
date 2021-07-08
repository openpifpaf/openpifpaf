#include <math.h>

#include <algorithm>

#include "openpifpaf/decoder/utils/nms_detections.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSDetections::suppression = 0.1;
double NMSDetections::suppression_soft = 0.3;
double NMSDetections::instance_threshold = 0.15;
double NMSDetections::iou_threshold = 0.7;
double NMSDetections::iou_threshold_soft = 0.5;


struct AnnotationCompare {
    bool operator() (const Detection& a, const Detection& b) {
        return (a.v > b.v);
    }
};


void NMSDetections::call(Occupancy* occupancy, std::vector<Detection>* annotations) {
    occupancy->clear();
    std::sort(annotations->begin(), annotations->end(), AnnotationCompare());

    // remove annotations below instance threshold
    annotations->erase(
        std::remove_if(annotations->begin(), annotations->end(), [=](const Detection& ann) {
            return (ann.v < instance_threshold);
        }),
        annotations->end()
    );

    std::sort(annotations->begin(), annotations->end(), AnnotationCompare());
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
