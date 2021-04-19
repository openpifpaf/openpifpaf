#include <torch/extension.h>

#include "openpifpaf.hpp"



TORCH_LIBRARY(my_ops, m) {
    m.def("cif_hr_accumulate_op", openpifpaf::decoder::utils::cif_hr_accumulate_op);
}


TORCH_LIBRARY(my_classes, m) {
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
  ;

  m.class_<openpifpaf::decoder::utils::CifHr>("CifHr")
    .def(torch::init<const at::IntArrayRef&, int64_t>())
    .def("accumulate", &openpifpaf::decoder::utils::CifHr::accumulate)
    .def("get_accumulated", &openpifpaf::decoder::utils::CifHr::get_accumulated)
    .def("reset", &openpifpaf::decoder::utils::CifHr::reset)
  ;

  m.class_<openpifpaf::decoder::utils::CifSeeds>("CifSeeds")
    .def(torch::init<const torch::Tensor&, double>())
    .def("fill", &openpifpaf::decoder::utils::CifSeeds::fill)
    .def("get", &openpifpaf::decoder::utils::CifSeeds::get)
  ;
  m.def("CifSeeds_set_threshold", openpifpaf::decoder::utils::CifSeeds::set_threshold);

  m.class_<openpifpaf::decoder::utils::CafScored>("CafScored")
    .def(torch::init<const torch::Tensor&, double, double, double>())
    .def("fill", &openpifpaf::decoder::utils::CafScored::fill)
    .def("get", &openpifpaf::decoder::utils::CafScored::get)
  ;

  m.class_<openpifpaf::decoder::CifCaf>("CifCaf")
    .def(torch::init<
        int64_t,
        std::vector<std::vector<int64_t> >
    >())
    .def("call", &openpifpaf::decoder::CifCaf::call)
  ;
  m.def("grow_connection_blend", openpifpaf::decoder::grow_connection_blend_py);
}
