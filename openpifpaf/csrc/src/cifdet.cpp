#include <algorithm>
#include <cmath>
#include <queue>

#include "openpifpaf/decoder/cifdet.hpp"

#include "openpifpaf/decoder/utils/cif_hr.hpp"
#include "openpifpaf/decoder/utils/cif_seeds.hpp"
#include "openpifpaf/decoder/utils/nms_detections.hpp"
#include "openpifpaf/decoder/utils/occupancy.hpp"


namespace openpifpaf {
namespace decoder {


int64_t CifDet::max_detections_before_nms = 120;


torch::Tensor CifDet::call(
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
    std::vector<Detection> annotations;

    int64_t f;
    float c, x, y, w, h;
    for (int64_t seed_i=0; seed_i < seeds_f.size(0); seed_i++) {
        f = seeds_f_a[seed_i];
        c = seeds_vxywh_a[seed_i][0];
        x = seeds_vxywh_a[seed_i][1];
        y = seeds_vxywh_a[seed_i][2];
        w = seeds_vxywh_a[seed_i][3];
        h = seeds_vxywh_a[seed_i][4];
        if (occupancy.get(f, x, y)) continue;

        occupancy.set(f, x, y, 0.1 * fmin(w, h));
        annotations.push_back(Detection(f, c, x - 0.5 * w, y - 0.5 * h, w, h));

        if (annotations.size() > max_detections_before_nms) break;
    }

#ifdef DEBUG
    TORCH_WARN("NMS");
#endif
    utils::NMSDetections().call(&occupancy, &annotations);

#ifdef DEBUG
    TORCH_WARN("convert to tensor");
#endif
    auto out = torch::zeros({ int64_t(annotations.size()), 5 });
    auto out_a = out.accessor<float, 2>();
    for (int64_t ann_i = 0; ann_i < int64_t(annotations.size()); ann_i++) {
        auto& ann = annotations[ann_i];
    }
    return out;
}


}  // namespace decoder
}  // namespace openpifpaf
