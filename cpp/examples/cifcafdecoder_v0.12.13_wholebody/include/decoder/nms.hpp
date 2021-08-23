#ifndef NMS_HPP
#define NMS_HPP

#include "field_config.hpp"
#include "annotation.hpp"

class Keypoints {
    public:
        float suppression = 0.0;
        float instance_threshold = 0.0;
        float keypoint_threshold = 0.0;

    public:
        vector<Annotation> annotations(vector<Annotation> &anns);
};

#endif