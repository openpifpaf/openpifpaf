#include <torch/extension.h>

#include "examples/cifcafdecoder/occupancy.hpp"


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
    .def(torch::init<double, double>())
    // // The next line registers a stateless (i.e. no captures) C++ lambda
    // // function as a method. Note that a lambda function must take a
    // // `c10::intrusive_ptr<YourClass>` (or some const/ref version of that)
    // // as the first argument. Other arguments can be whatever you want.
    // .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>& self) {
    //   return self->stack_.back();
    // })
    // // The following four lines expose methods of the MyStackClass<std::string>
    // // class as-is. `torch::class_` will automatically examine the
    // // argument and return types of the passed-in method pointers and
    // // expose these to Python and TorchScript accordingly. Finally, notice
    // // that we must take the *address* of the fully-qualified method name,
    // // i.e. use the unary `&` operator, due to C++ typing rules.
    // .def("push", &MyStackClass<std::string>::push)
    // .def("pop", &MyStackClass<std::string>::pop)
    // .def("clone", &MyStackClass<std::string>::clone)
    // .def("merge", &MyStackClass<std::string>::merge)
  ;
}
