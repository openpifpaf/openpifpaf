#ifndef NUMPY_UTILS_HPP
#define NUMPY_UTILS_HPP

#include "../../include/decoder/field_config.hpp"

float maximum1d(const vector<float> v);
float minimum1d(const vector<float> v);

float norm2d(const Vector2D &v);
vector<float> norm2d_axis0(const Vector2D &v);
vector<float> maximum(float threshold, const vector<float> &array);
Vector3D maximum3d(const Vector3D &a, const Vector3D &b);
Vector2D subtract2d(const Vector2D &v1, const Vector2D &v2);

vector<bool> mask1d(const vector<float> &v, float threshold);
vector<bool> mask1d_lt(const vector<float> &v, float threshold);

blobnd<bool> mask2d(
    blobnd<float> &v, int index_d4, int index_d3,
    float filter_value
);
Vector2D filter_vector3D(
    blobnd<float> &v, int index_d4, int filter_index,
    blobnd<bool> &mask
);

vector<float> filter1d(const vector<float> &v, const vector<bool> &mask);
Vector2D filter_vector2D(const Vector2D &v, int filter_index, float filter_value);
Vector2D filter_vector2D(const Vector2D &v, const vector<bool> &mask);

vector<float> full1d(int size, float val);

vector<float> vector1d_ones(int shape);
vector<float> vector1d_zeros(int shape);
Vector2D vector2d_zeros(const vector<int> &shape);
Vector3D vector3d_zeros(const vector<int> &shape);
Vector4D vector4d_zeros(const vector<int> &shape);

#endif