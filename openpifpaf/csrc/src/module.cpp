#include <torch/script.h>

#include "openpifpaf.hpp"


// Win32 needs this.
#ifdef _WIN32
#include <Python.h>
PyMODINIT_FUNC PyInit__cpp(void) {
    return NULL;
}
#endif


TORCH_LIBRARY(openpifpaf_decoder, m) {
    m.class_<openpifpaf::decoder::CifCaf>("CifCaf")
        .def_static("set_block_joints", &openpifpaf::decoder::CifCaf::set_block_joints)
        .def_static("get_block_joints", &openpifpaf::decoder::CifCaf::get_block_joints)
        .def_static("set_greedy", &openpifpaf::decoder::CifCaf::set_greedy)
        .def_static("get_greedy", &openpifpaf::decoder::CifCaf::get_greedy)
        .def_static("set_keypoint_threshold", &openpifpaf::decoder::CifCaf::set_keypoint_threshold)
        .def_static("get_keypoint_threshold", &openpifpaf::decoder::CifCaf::get_keypoint_threshold)
        .def_static("set_keypoint_threshold_rel", &openpifpaf::decoder::CifCaf::set_keypoint_threshold_rel)
        .def_static("get_keypoint_threshold_rel", &openpifpaf::decoder::CifCaf::get_keypoint_threshold_rel)
        .def_static("set_reverse_match", &openpifpaf::decoder::CifCaf::set_reverse_match)
        .def_static("get_reverse_match", &openpifpaf::decoder::CifCaf::get_reverse_match)
        .def_static("set_force_complete", &openpifpaf::decoder::CifCaf::set_force_complete)
        .def_static("get_force_complete", &openpifpaf::decoder::CifCaf::get_force_complete)
        .def_static("set_force_complete_caf_th", &openpifpaf::decoder::CifCaf::set_force_complete_caf_th)
        .def_static("get_force_complete_caf_th", &openpifpaf::decoder::CifCaf::get_force_complete_caf_th)

        .def(torch::init<int64_t, const torch::Tensor&>())
        .def("call", &openpifpaf::decoder::CifCaf::call)
    ;
    m.def("grow_connection_blend", openpifpaf::decoder::grow_connection_blend_py);
    m.def("cifcaf_op", openpifpaf::decoder::cifcaf_op);

    m.class_<openpifpaf::decoder::CifDet>("CifDet")
        .def_static("set_max_detections_before_nms", &openpifpaf::decoder::CifDet::set_max_detections_before_nms)
        .def_static("get_max_detections_before_nms", &openpifpaf::decoder::CifDet::get_max_detections_before_nms)

        .def(torch::init<>())
        .def("call", &openpifpaf::decoder::CifDet::call)
    ;
}


TORCH_LIBRARY(openpifpaf, m) {
    m.def("cif_hr_accumulate_op", openpifpaf::decoder::utils::cif_hr_accumulate_op);

    m.class_<openpifpaf::decoder::utils::Occupancy>("Occupancy")
        .def(torch::init<double, double>())
        .def("get", &openpifpaf::decoder::utils::Occupancy::get)
        .def("set", &openpifpaf::decoder::utils::Occupancy::set)
        .def("reset", &openpifpaf::decoder::utils::Occupancy::reset)
        .def("clear", &openpifpaf::decoder::utils::Occupancy::clear)
    ;

    m.class_<openpifpaf::decoder::utils::CifHr>("CifHr")
        .def_static("set_neighbors", &openpifpaf::decoder::utils::CifHr::set_neighbors)
        .def_static("get_neighbors", &openpifpaf::decoder::utils::CifHr::get_neighbors)
        .def_static("set_threshold", &openpifpaf::decoder::utils::CifHr::set_threshold)
        .def_static("get_threshold", &openpifpaf::decoder::utils::CifHr::get_threshold)

        .def(torch::init<>())
        .def("accumulate", &openpifpaf::decoder::utils::CifHr::accumulate)
        .def("get_accumulated", &openpifpaf::decoder::utils::CifHr::get_accumulated)
        .def("reset", &openpifpaf::decoder::utils::CifHr::reset)
    ;

    m.class_<openpifpaf::decoder::utils::CifSeeds>("CifSeeds")
        .def_static("set_threshold", &openpifpaf::decoder::utils::CifSeeds::set_threshold)
        .def_static("get_threshold", &openpifpaf::decoder::utils::CifSeeds::get_threshold)

        .def(torch::init<const torch::Tensor&, double>())
        .def("fill", &openpifpaf::decoder::utils::CifSeeds::fill)
        .def("get", &openpifpaf::decoder::utils::CifSeeds::get)
    ;

    m.class_<openpifpaf::decoder::utils::CafScored>("CafScored")
        .def_static("set_default_score_th", &openpifpaf::decoder::utils::CafScored::set_default_score_th)
        .def_static("get_default_score_th", &openpifpaf::decoder::utils::CafScored::get_default_score_th)

        .def(torch::init<const torch::Tensor&, double, double, double>())
        .def("fill", &openpifpaf::decoder::utils::CafScored::fill)
        .def("get", &openpifpaf::decoder::utils::CafScored::get)
    ;

    m.class_<openpifpaf::decoder::utils::NMSKeypoints>("NMSKeypoints")
        .def_static("set_instance_threshold", &openpifpaf::decoder::utils::NMSKeypoints::set_instance_threshold)
        .def_static("get_instance_threshold", &openpifpaf::decoder::utils::NMSKeypoints::get_instance_threshold)
        .def_static("set_keypoint_threshold", &openpifpaf::decoder::utils::NMSKeypoints::set_keypoint_threshold)
        .def_static("get_keypoint_threshold", &openpifpaf::decoder::utils::NMSKeypoints::get_keypoint_threshold)
        .def_static("set_suppression", &openpifpaf::decoder::utils::NMSKeypoints::set_suppression)
        .def_static("get_suppression", &openpifpaf::decoder::utils::NMSKeypoints::get_suppression)
    ;

    m.class_<openpifpaf::decoder::utils::NMSDetections>("NMSDetections")
        .def_static("set_instance_threshold", &openpifpaf::decoder::utils::NMSDetections::set_instance_threshold)
        .def_static("get_instance_threshold", &openpifpaf::decoder::utils::NMSDetections::get_instance_threshold)
        .def_static("set_iou_threshold", &openpifpaf::decoder::utils::NMSDetections::set_iou_threshold)
        .def_static("get_iou_threshold", &openpifpaf::decoder::utils::NMSDetections::get_iou_threshold)
        .def_static("set_iou_threshold_soft", &openpifpaf::decoder::utils::NMSDetections::set_iou_threshold_soft)
        .def_static("get_iou_threshold_soft", &openpifpaf::decoder::utils::NMSDetections::get_iou_threshold_soft)
        .def_static("set_suppression", &openpifpaf::decoder::utils::NMSDetections::set_suppression)
        .def_static("get_suppression", &openpifpaf::decoder::utils::NMSDetections::get_suppression)
        .def_static("set_suppression_soft", &openpifpaf::decoder::utils::NMSDetections::set_suppression_soft)
        .def_static("get_suppression_soft", &openpifpaf::decoder::utils::NMSDetections::get_suppression_soft)
    ;
}
