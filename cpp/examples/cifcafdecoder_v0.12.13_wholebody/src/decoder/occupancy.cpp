#include <cassert>
#include <math.h>

#include "../../include/decoder/occupancy.hpp"
#include "../../include/decoder/functional.hpp"
#include "../../include/utils/numpy_utils.hpp"


Occupancy::Occupancy(const vector<int> &shape, float reduction, float min_scale) {
    assert(shape.size() == 3 && "Shape must be 3");

    if(min_scale == -1.0)
        min_scale = reduction;

    assert(min_scale >= reduction && "min_scale must be greater or equal to reduction");

    this->reduction = reduction;
    this->min_scale = min_scale;
    this->min_scale_reduced = min_scale / reduction;

    int d1 = shape[0];
    int d2 = (int)(shape[1] / reduction);
    int d3 = (int)(shape[2] / reduction);
    vector<int> d {d1, d2, d3};

    this->occupancy = vector3d_zeros(d);
}

void Occupancy::set(int f, float x, float y, float sigma) {
    if(f >= this->occupancy.size())
        return;

    float xi = round(x / this->reduction);
    float yi = round(y / this->reduction);
    float si = round(max(this->min_scale_reduced, sigma / this->reduction));

    scalar_square_add_single(this->occupancy[f], xi, yi, si, 1);
}

float Occupancy::get(int f, float x, float y) {
    if(f >= this->occupancy.size())
        return 1.0;

    return scalar_nonzero_clipped_with_reduction(this->occupancy[f], x, y, this->reduction);
}