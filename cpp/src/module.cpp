#include <torch/script.h>

#include "openpifpaf.hpp"


TORCH_LIBRARY(openpifpaf, m) {
  m.def("cif_hr_accumulate_op", openpifpaf::decoder::utils::cif_hr_accumulate_op);

  m.class_<openpifpaf::decoder::utils::Occupancy>("Occupancy")
    .def(torch::init<double, double>())
    // // The next line registers a stateless (i.e. no captures) C++ lambda
    // // function as a method. Note that a lambda function must take a
    // // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // // as the first argument. Other arguments can be whatever you want.
    // .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
    //   return self->stack_.back();
    // })
    .def("get", &openpifpaf::decoder::utils::Occupancy::get)
    .def("set", &openpifpaf::decoder::utils::Occupancy::set)
    .def("reset", &openpifpaf::decoder::utils::Occupancy::reset)
    .def("clear", &openpifpaf::decoder::utils::Occupancy::clear)
  ;

  m.class_<openpifpaf::decoder::utils::CifHr>("CifHr")
    .def(torch::init<>())
    .def("accumulate", &openpifpaf::decoder::utils::CifHr::accumulate)
    .def("get_accumulated", &openpifpaf::decoder::utils::CifHr::get_accumulated)
    .def("reset", &openpifpaf::decoder::utils::CifHr::reset)
  ;

  m.class_<openpifpaf::decoder::utils::CifSeeds>("CifSeeds")
    .def(torch::init<const torch::Tensor&, double>())
    .def("fill", &openpifpaf::decoder::utils::CifSeeds::fill)
    .def("get", &openpifpaf::decoder::utils::CifSeeds::get)
  ;
  m.def("CifSeeds_threshold", openpifpaf::decoder::utils::CifSeeds::set_threshold);

  m.class_<openpifpaf::decoder::utils::CafScored>("CafScored")
    .def(torch::init<const torch::Tensor&, double, double, double>())
    .def("fill", &openpifpaf::decoder::utils::CafScored::fill)
    .def("get", &openpifpaf::decoder::utils::CafScored::get)
  ;

  m.class_<openpifpaf::decoder::CifCaf>("CifCaf")
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
    .def(torch::init<
        int64_t,
        std::vector<std::vector<int64_t> >
    >())
    .def("call", &openpifpaf::decoder::CifCaf::call)
  ;
  m.def("grow_connection_blend", openpifpaf::decoder::grow_connection_blend_py);
}
