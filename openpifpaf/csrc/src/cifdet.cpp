#include <algorithm>
#include <cmath>
#include <queue>

#include "openpifpaf/decoder/cifdet.hpp"

#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/cif_seeds.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {


int64_t CifDet::max_detections_before_nms = 120;


struct BBox {
    float x, y, w, h;
};


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CifDet::call(
    const torch::Tensor& cifdet_field,
    int64_t cifdet_stride
) {
    TORCH_CHECK(cifdet_field.device().is_cpu(), "cifdet_field must be a CPU tensor");

    cifDetHr.reset(cifdet_field.sizes(), cifdet_stride);
    cifDetHr.accumulate(cifdet_field, cifdet_stride, 0.0, 1.0);
    auto [cifDetHr_accumulated, cifDetHr_revision] = cifDetHr.get_accumulated();

    utils::CifDetSeeds seeds(cifDetHr_accumulated, cifDetHr_revision);
    seeds.fill(cifdet_field, cifdet_stride);
    auto [seeds_f, seeds_vxywh] = seeds.get();
    auto seeds_f_a = seeds_f.accessor<int64_t, 1>();
    auto seeds_vxywh_a = seeds_vxywh.accessor<float, 2>();
#ifdef DEBUG
    TORCH_WARN("seeds=", seeds_f_a.size(0));
#endif

    occupancy.reset(cifDetHr_accumulated.sizes());

    int64_t f;
    float c, x, y, w, h;
    std::vector<int64_t> categories;
    std::vector<float> scores;
    std::vector<BBox> boxes;
    for (int64_t seed_i=0; seed_i < seeds_f.size(0); seed_i++) {
        f = seeds_f_a[seed_i];
        c = seeds_vxywh_a[seed_i][0];
        x = seeds_vxywh_a[seed_i][1];
        y = seeds_vxywh_a[seed_i][2];
        w = seeds_vxywh_a[seed_i][3];
        h = seeds_vxywh_a[seed_i][4];
        if (occupancy.get(f, x, y)) continue;

        occupancy.set(f, x, y, 0.1 * fmin(w, h));
        categories.push_back(f + 1);
        scores.push_back(c);
        boxes.push_back({ x - 0.5f * w, y - 0.5f * h, x + 0.5f * w, y + 0.5f * h });

        if (int64_t(boxes.size()) >= max_detections_before_nms) break;
    }

#ifdef DEBUG
    TORCH_WARN("convert to tensor");
#endif
    auto categories_t = torch::from_blob(
        reinterpret_cast<int64_t*>(categories.data()),
        categories.size(), torch::kInt64
    ).clone();
    auto scores_t = torch::from_blob(
        reinterpret_cast<float*>(scores.data()), scores.size()).clone();
    auto boxes_t = torch::from_blob(
        reinterpret_cast<float*>(boxes.data()), { int64_t(boxes.size()), 4}).clone();

    return std::make_tuple(categories_t, scores_t, boxes_t);
}


}  // namespace decoder
}  // namespace openpifpaf
