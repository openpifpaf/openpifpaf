#pragma once

#include <torch/script.h>


namespace openpifpaf {
namespace decoder {
namespace utils {


struct CompositeAssociation {
    float c, x1, y1, b1, s1, x2, y2, b2, s2;

    CompositeAssociation(
        float c_,
        float x1_, float y1_, float b1_, float s1_,
        float x2_, float y2_, float b2_, float s2_
    ) :
        c(c_),
        x1(x1_), y1(y1_), b1(b1_), s1(s1_),
        x2(x2_), y2(y2_), b2(b2_), s2(s2_)
    { }
};


struct CafScored : torch::CustomClassHolder {
    torch::TensorAccessor<float, 3UL> cifhr_a;
    static double default_score_th;
    double score_th;
    double cif_floor;

    std::vector<CompositeAssociation> forward;
    std::vector<CompositeAssociation> backward;

    CafScored(
        const torch::Tensor& cifhr_,
        double score_th_=-1.0,
        double cif_floor_=0.1
    ) :
        cifhr_a(cifhr_.accessor<float, 3>()),
        score_th(score_th_ >= 0.0 ? score_th_ : default_score_th),
        cif_floor(cif_floor_)
    { }
    void fill(const torch::Tensor& caf_field, int64_t stride);
    std::tuple<torch::Tensor, torch::Tensor> get(void);

    float cifhr_value(int64_t f, float x, float y, float default_value=-1.0);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
