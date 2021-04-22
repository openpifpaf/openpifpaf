#pragma once

#include <torch/script.h>

#include "openpifpaf/utils.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


struct CompositeAssociation {
    float c, x1, y1, x2, y2, b1, b2, s1, s2;

    CompositeAssociation() :
        c(0.0),
        x1(0.0), y1(0.0), x2(0.0), y2(0.0),
        b1(0.0), b2(0.0),
        s1(0.0), s2(0.0) { }
    CompositeAssociation(
        float c_,
        float x1_, float y1_, float x2_, float y2_,
        float b1_, float b2_,
        float s1_, float s2_
    ) :
        c(c_),
        x1(x1_), y1(y1_), x2(x2_), y2(y2_),
        b1(b1_), b2(b2_),
        s1(s1_), s2(s2_)
    { }
};


struct CafScored : torch::CustomClassHolder {
    torch::TensorAccessor<float, 3UL> cifhr_a;
    double cifhr_revision;
    double score_th;
    double cif_floor;

    std::vector<std::vector<CompositeAssociation> > forward;
    std::vector<std::vector<CompositeAssociation> > backward;

    static double default_score_th;

    STATIC_GETSET(double, default_score_th)

    CafScored(
        const torch::Tensor& cifhr_,
        double cifhr_revision_,
        double score_th_=-1.0,
        double cif_floor_=0.1
    ) :
        cifhr_a(cifhr_.accessor<float, 3>()),
        cifhr_revision(cifhr_revision_),
        score_th(score_th_ >= 0.0 ? score_th_ : default_score_th),
        cif_floor(cif_floor_)
    { }
    void fill(const torch::Tensor& caf_field, int64_t stride, const torch::Tensor& skeleton);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor> > get(void);

    float cifhr_value(int64_t f, float x, float y, float default_value=-1.0);
};


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
