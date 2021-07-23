#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include "torch/script.h"


int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: openpifpaf-example <path-to-exported-script-module> <path-to-png-image>\n";
        return -1;
    }

    // load model from file
    torch::jit::script::Module model = torch::jit::load(argv[1]);

    // load image file into tensor
    cv::Mat frame = cv::imread(argv[2], 1);
    cv::Size frame_size = frame.size();
    std::cout << "frame size: " << frame_size.height << ", " << frame_size.width << std::endl;
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32FC3, 4.0f / 255.0f, -2.0);  // color range: [-2, 2]
    auto input_tensor = torch::from_blob(frame.data, {1, frame_size.height, frame_size.width, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    std::cout << "input tensor sizes: " << input_tensor.sizes() << std::endl;

    // create model inputs and execute forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    auto output = model.forward(inputs).toList();
    std::cout << "output: " << output << '\n';
}
