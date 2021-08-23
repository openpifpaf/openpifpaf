#ifndef OCCUPANCY_HPP
#define OCCUPANCY_HPP

#include "field_config.hpp"

class Occupancy {
    public:
        float reduction;
        float min_scale;
        float min_scale_reduced;
        Vector3D occupancy;

    public:
        Occupancy() {}
        Occupancy(const vector<int> &shape, float reduction, float min_scale);

        void set(int f, float x, float y, float sigma);
        float get(int f, float x, float y); // return 0.0 or 1.0
};

#endif