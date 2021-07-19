#include <torch/script.h>

#include <iostream>
#include <memory>


int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: openpifpaf-example <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}
