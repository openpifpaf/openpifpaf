#include <torch/extension.h>

#include "openpifpaf.hpp"


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     // py::module decoder = m.def_submodule("decoder");
//     // py::module decoder_utils = decoder.def_submodule("utils");
//     torch::python::bind_module<decoder::utils::Occupancy>(m, "Occupancy");
//         // .def(py::init<int, int>())
//         // .def("forward", &decoder::utils::Occupancy::forward);
//     // torch::python::add_module_bindings(
//     //     py::class_<decoder::utils::Occupancy, std::shared_ptr<decoder::utils::Occupancy>>(decoder_utils, "Occupancy")
//     // );
// }


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
}
