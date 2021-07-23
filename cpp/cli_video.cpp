#include <iostream>
#include <memory>

#include "opencv2/opencv.hpp"
#include "torch/script.h"


int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: openpifpaf-example <path-to-exported-script-module> <path-to-video> [width] [height]\n";
        return -1;
    }

    // load model from file
    torch::jit::script::Module model = torch::jit::load(argv[1]);

    // load video
    cv::VideoCapture cap;
    if (strlen(argv[2]) == 1) {
        cap = cv::VideoCapture(atoi(argv[2]));
    } else {
        cap = cv::VideoCapture(argv[2]);
    }

    cv::Mat frame, frame_normalized;
    while (cap.isOpened()) {
        auto start = std::chrono::high_resolution_clock::now();
        cap >> frame;

        if (frame.empty()) {
            std::cout << "No new frame available.\n";
            break;
        }

        // reformat OpenCV frame into PyTorch tensor
        if (argc == 5) cv::resize(frame, frame, {atoi(argv[3]), atoi(argv[4])});
        cv::Size frame_size = frame.size();
        std::cout << "frame size: " << frame_size.height << ", " << frame_size.width << std::endl;
        cv::cvtColor(frame, frame_normalized, cv::COLOR_BGR2RGB);
        frame_normalized.convertTo(frame_normalized, CV_32FC3, 4.0f / 255.0f, -2.0);  // color range: [-2, 2]
        auto input_tensor = torch::from_blob(frame_normalized.data, {1, frame_size.height, frame_size.width, 3});
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        std::cout << "input tensor sizes: " << input_tensor.sizes() << std::endl;

        // create model inputs and execute forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        auto output = model.forward(inputs).toList();
        auto output_0 = output.get(0).toTuple();
        auto joints = output_0->elements()[0].toTensor();

        // show
        auto joints_a = joints.accessor<float, 3>();
        for (int64_t i=0; i < joints.size(0); i++) {
            for (int64_t j=0; j < joints.size(1); j++) {
                if (joints_a[i][j][0] < 0.2) continue;
                cv::circle(
                    frame,
                    {static_cast<int>(joints_a[i][j][1]), static_cast<int>(joints_a[i][j][2])},
                    2,
                    cv::Scalar(255, 255, 255),
                    cv::FILLED
                );
            }
        }
        cv::imshow("preview", frame);

        // timing
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        int64_t microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "FPS: " << 1e6 / microseconds << std::endl;

        // break on ESC key
        if (cv::waitKey(1) == 27) break;
    }
}
