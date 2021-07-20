#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include "torch/script.h"


int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: openpifpaf-example <path-to-exported-script-module> <path-to-png-image>\n";
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

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1, 3, 193, 193}));

    cv::Mat frame = cv::imread(argv[2], 1);
    cv::Size frame_size = frame.size();
    std::cout << "frame size: " << frame_size.height << ", " << frame_size.width << std::endl;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    auto input_tensor = torch::from_blob(frame.data, {1, frame_size.height, frame_size.width, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    std::cout << "input tensor sizes: " << input_tensor.sizes() << std::endl;
    inputs.push_back((input_tensor - 0.5) * 4.0);

    // Execute the model and turn its output into a tensor.
    auto output = module.forward(inputs).toList();
    std::cout << "output: " << output << '\n';

    std::cout << "ok\n";
}
