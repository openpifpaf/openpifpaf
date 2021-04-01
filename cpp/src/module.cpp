#include <torch/extension.h>

#include "openpifpaf.hpp"



TORCH_LIBRARY(my_ops, m) {
    m.def("cif_hr_accumulate_op", openpifpaf::decoder::utils::cif_hr_accumulate_op);
}


TORCH_LIBRARY(my_classes, m) {
  m.class_<openpifpaf::decoder::utils::Occupancy>("Occupancy")
    .def(torch::init<const at::IntArrayRef&, double, double>())
    // // The next line registers a stateless (i.e. no captures) C++ lambda
    // // function as a method. Note that a lambda function must take a
    // // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // // as the first argument. Other arguments can be whatever you want.
    // .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
    //   return self->stack_.back();
    // })
    .def("get", &openpifpaf::decoder::utils::Occupancy::get)
    .def("set", &openpifpaf::decoder::utils::Occupancy::set)
  ;

  m.class_<openpifpaf::decoder::utils::CifHr>("CifHr")
    .def(torch::init<const at::IntArrayRef&, int64_t>())
    .def("accumulate", &openpifpaf::decoder::utils::CifHr::accumulate)
  ;
}
