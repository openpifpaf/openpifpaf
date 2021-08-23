#include <algorithm>
#include <cassert>

#include "../../include/decoder/annotation.hpp"
#include "../../include/decoder/occupancy.hpp"
#include "../../include/utils/numpy_utils.hpp"
#include "../../include/decoder/nms.hpp"


bool compare_annotations(Annotation &a1, Annotation &a2) {
    return a1.score() > a2.score();
}

vector<Annotation> Keypoints::annotations(vector<Annotation> &anns) {
    for(int i = 0; i < anns.size(); i++) {
        vector<float> v;
        for(int j = 0; j < anns[i].data.size(); j++) {
            v.push_back(anns[i].data[j][2]);
        }
        vector<bool> mask = mask1d_lt(v, this->keypoint_threshold);

        for(int k = 0; k < mask.size(); k++) {
            if(mask[k]) {
                for(int l = 0; l < anns[i].data[0].size(); l++) {
                    anns[i].data[k][l] = 0.0;
                }
            }
        }
    }

    vector<Annotation> filtered_annotations;
    for(int i = 0; i < anns.size(); i++) {
        if(anns[i].score() >= this->instance_threshold)
            filtered_annotations.push_back(anns[i]);
    }
    if(filtered_annotations.empty())
        return filtered_annotations;

    int d1 = filtered_annotations[0].data.size();
    float d2;
    float d3;
    for(int i = 0; i < filtered_annotations.size(); i++) {
        vector<float> v2;
        vector<float> v3;
        for(int j = 0; j <  filtered_annotations[i].data.size(); j++) {
            v2.push_back(filtered_annotations[i].data[j][1]);
            v3.push_back(filtered_annotations[i].data[j][0]);
        }
        float max_v2 = maximum1d(v2);
        float max_v3 = maximum1d(v3);
        if(i == 0) {
            d2 = max_v2;
            d3 = max_v3;
        } else {
            if(max_v2 > d2)
                d2 = max_v2;
            if(max_v3 > d3)
                d3 = max_v3;
        }
    }

    vector<int> shape = {d1, (int)(d2 + 1), (int)(d3 + 1)};
    Occupancy occupied = Occupancy(shape, 2.0, 4.0);

    sort(filtered_annotations.begin(), filtered_annotations.end(), compare_annotations);
    for(int i = 0; i < filtered_annotations.size(); i++) {
        assert(!filtered_annotations[i].joint_scales.empty() && "Annotation join scales must not be empty");
        assert(occupied.occupancy.size() == filtered_annotations[i].data.size() && "Occupied and annotation data size must match");

        for(int j = 0; j < filtered_annotations[i].data.size(); j++) {
            vector<float> xyv = filtered_annotations[i].data[j];
            float joint_s = filtered_annotations[i].joint_scales[j];
            float v = xyv[2];
            if(v == 0.0)
                continue;

            if(occupied.get(j, xyv[0], xyv[1])) {
                xyv[2] *= this->suppression;
            } else {
                occupied.set(j, xyv[0], xyv[1], joint_s);
            }
        }
    }

    for(int i = 0; i < filtered_annotations.size(); i++) {
        vector<float> v;
        for(int j = 0; j < filtered_annotations[i].data.size(); j++) {
            v.push_back(filtered_annotations[i].data[j][2]);
        }
        vector<bool> mask = mask1d_lt(v, this->keypoint_threshold);

        for(int k = 0; k < mask.size(); k++) {
            if(mask[k]) {
                for(int l = 0; l < filtered_annotations[i].data[0].size(); l++) {
                    filtered_annotations[i].data[k][l] = 0.0;
                }
            }
        }
    }

    vector<Annotation> filtered_annotations_by_score;
    for(int i = 0; i < filtered_annotations.size(); i++) {
        if(filtered_annotations[i].score() >= this->instance_threshold)
            filtered_annotations_by_score.push_back(filtered_annotations[i]);
    }

    sort(filtered_annotations_by_score.begin(), filtered_annotations_by_score.end(), compare_annotations);
    return filtered_annotations_by_score;
}