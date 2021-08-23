#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include "field_config.hpp"


float clip(float a, float min_value, float max_value);
float approx_exp(float x);

void scalar_square_add_gauss_with_max(
    blobnd<float> &field, int field_index,
    const vector<float> &x,
    const vector<float> &y,
    const vector<float> &sigma,
    const vector<float> &v,
    float truncate=2.0,
    float max_value=1.0
);

vector<float> scalar_values(
    blobnd<float> &field, int field_index,
    const vector<float> &x,
    const vector<float> &y,
    float default_val=-1,
    int debug=0
);

vector<float> grow_connection_blend(
    const Vector2D &caf_field,
    float x,
    float y,
    float xy_scale,
    bool only_max = false
);

float scalar_nonzero_clipped_with_reduction(const Vector2D &field, float x, float y, float r);
void scalar_square_add_single(Vector2D &field, float x, float y, float sigma, float value);

#endif