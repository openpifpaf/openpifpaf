#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct Seed {
    int64_t f;
    float v, x, y, s;

    Seed(int64_t f_, float v_, float x_, float y_, float s_)
        : f(f_), v(v_), x(x_), y(y_), s(s_) { }
};


struct CifSeeds : torch::CustomClassHolder {
    torch::TensorAccessor<float, 3UL> cifhr_a;
    double cifhr_revision;
    std::vector<Seed> seeds;
    static double v_threshold;

    static void set_threshold(double v) { v_threshold = v; }

    CifSeeds(const torch::Tensor& cifhr_, double cifhr_revision_)
    : cifhr_a(cifhr_.accessor<float, 3>()),
      cifhr_revision(cifhr_revision_)
    { }
    void fill(const torch::Tensor& cif_field, int64_t stride);
    std::tuple<torch::Tensor, torch::Tensor> get(void);

    float cifhr_value(int64_t f, float x, float y, float default_value=-1.0);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
