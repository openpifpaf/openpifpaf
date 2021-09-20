#include <torch/script.h>

#include "openpifpaf.hpp"


// Win32 needs this.
#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit__cpp(void) {
    return NULL;
}
#endif


// TODO the following is a temporary workaround for https://github.com/pytorch/pytorch/issues/56571
#define STATIC_GETSET(C, T, V) .def_static("set_"#V, [](T v) { C = v; }).def_static("get_"#V, []() { return C; })


TORCH_LIBRARY(openpifpaf, m) {
    m.def("set_quiet", openpifpaf::set_quiet);
}


TORCH_LIBRARY(openpifpaf_decoder, m) {
    m.class_<openpifpaf::decoder::CifCaf>("CifCaf")
        STATIC_GETSET(openpifpaf::decoder::CifCaf::block_joints, bool, block_joints)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::greedy, bool, greedy)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::keypoint_threshold, double, keypoint_threshold)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::keypoint_threshold_rel, double, keypoint_threshold_rel)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::reverse_match, bool, reverse_match)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::force_complete, bool, force_complete)
        STATIC_GETSET(openpifpaf::decoder::CifCaf::force_complete_caf_th, double, force_complete_caf_th)

        .def(torch::init<int64_t, const torch::Tensor&>())
        .def("call", &openpifpaf::decoder::CifCaf::call)
        .def("call_with_initial_annotations", &openpifpaf::decoder::CifCaf::call_with_initial_annotations)
        .def("get_cifhr", [](const c10::intrusive_ptr<openpifpaf::decoder::CifCaf>& self) {
            return self->cifhr.get_accumulated();
        })

        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<openpifpaf::decoder::CifCaf>& self)
                    -> std::tuple<int64_t, torch::Tensor> {
                return std::make_tuple(self->n_keypoints, self->skeleton);
            },
            // __setstate__
            [](std::tuple<int64_t, torch::Tensor> state)
                    -> c10::intrusive_ptr<openpifpaf::decoder::CifCaf> {
                auto [n_keypoints, skeleton] = state;
                return c10::make_intrusive<openpifpaf::decoder::CifCaf>(n_keypoints, skeleton);
            }
        )
    ;
    m.def("grow_connection_blend", openpifpaf::decoder::grow_connection_blend_py);

    m.class_<openpifpaf::decoder::CifDet>("CifDet")
        STATIC_GETSET(openpifpaf::decoder::CifDet::max_detections_before_nms, int64_t, max_detections_before_nms)

        .def(torch::init<>())
        .def("call", &openpifpaf::decoder::CifDet::call)
    ;
}


TORCH_LIBRARY(openpifpaf_decoder_utils, m) {
    m.class_<openpifpaf::decoder::utils::Occupancy>("Occupancy")
        .def(torch::init<double, double>())
        .def("get", &openpifpaf::decoder::utils::Occupancy::get)
        .def("set", &openpifpaf::decoder::utils::Occupancy::set)
        .def("reset", &openpifpaf::decoder::utils::Occupancy::reset)
        .def("clear", &openpifpaf::decoder::utils::Occupancy::clear)
    ;

    m.class_<openpifpaf::decoder::utils::CifHr>("CifHr")
        STATIC_GETSET(openpifpaf::decoder::utils::CifHr::neighbors, int64_t, neighbors)
        STATIC_GETSET(openpifpaf::decoder::utils::CifHr::threshold, double, threshold)
        STATIC_GETSET(openpifpaf::decoder::utils::CifHr::ablation_skip, bool, ablation_skip)

        .def(torch::init<>())
        .def("accumulate", &openpifpaf::decoder::utils::CifHr::accumulate)
        .def("get_accumulated", &openpifpaf::decoder::utils::CifHr::get_accumulated)
        .def("reset", &openpifpaf::decoder::utils::CifHr::reset)
    ;

    m.class_<openpifpaf::decoder::utils::CifSeeds>("CifSeeds")
        STATIC_GETSET(openpifpaf::decoder::utils::CifSeeds::threshold, double, threshold)
        STATIC_GETSET(openpifpaf::decoder::utils::CifSeeds::ablation_nms, bool, ablation_nms)
        STATIC_GETSET(openpifpaf::decoder::utils::CifSeeds::ablation_no_rescore, bool, ablation_no_rescore)

        .def(torch::init<const torch::Tensor&, double>())
        .def("fill", &openpifpaf::decoder::utils::CifSeeds::fill)
        .def("get", &openpifpaf::decoder::utils::CifSeeds::get)
    ;

    m.class_<openpifpaf::decoder::utils::CifDetSeeds>("CifDetSeeds")
        STATIC_GETSET(openpifpaf::decoder::utils::CifDetSeeds::threshold, double, threshold)

        .def(torch::init<const torch::Tensor&, double>())
        .def("fill", &openpifpaf::decoder::utils::CifDetSeeds::fill)
        .def("get", &openpifpaf::decoder::utils::CifDetSeeds::get)
    ;

    m.class_<openpifpaf::decoder::utils::CafScored>("CafScored")
        STATIC_GETSET(openpifpaf::decoder::utils::CafScored::default_score_th, double, default_score_th)
        STATIC_GETSET(openpifpaf::decoder::utils::CafScored::ablation_no_rescore, bool, ablation_no_rescore)

        .def(torch::init<const torch::Tensor&, double, double, double>())
        .def("fill", &openpifpaf::decoder::utils::CafScored::fill)
        .def("get", &openpifpaf::decoder::utils::CafScored::get)
    ;

    m.class_<openpifpaf::decoder::utils::NMSKeypoints>("NMSKeypoints")
        STATIC_GETSET(openpifpaf::decoder::utils::NMSKeypoints::instance_threshold, double, instance_threshold)
        STATIC_GETSET(openpifpaf::decoder::utils::NMSKeypoints::keypoint_threshold, double, keypoint_threshold)
        STATIC_GETSET(openpifpaf::decoder::utils::NMSKeypoints::suppression, double, suppression)
    ;
}
