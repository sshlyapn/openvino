#include "ie_wrapper.hpp"
#include "cmd_params.hpp"

#include <chrono>
#include <thread>
#include <string>
#include <opencv2/imgcodecs.hpp>


using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    // ---------------------------------Parsing and validation of input
    // args----------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }

    std::cout << "Parsing input parameters" << std::endl;
    if (FLAGS_test_files.empty())
    {
      throw std::logic_error("Parameter -test_files is not set");
    }
    if (FLAGS_m_det.empty())
    {
        throw std::logic_error("Parameter -m_det is not set");
    }

    return true;
}

// void compareResults(const std::map<size_t, std::vector<std::vector<DetObject>>> &fisrt_results, const std::map<size_t, std::vector<std::vector<DetObject>>> &second_results) {
//     for (const auto &first_frame_results : fisrt_results) {
//         size_t image_id = first_frame_results.first;
//         std::cout << std::to_string(image_id) + " frame results comparision" << std::endl;
//         std::vector<DetObject> reference = first_frame_results.second[0];
//         for (const auto &infer_result : second_results.at(image_id)) {
//             if (reference.size() != infer_result.size())
//                 throw std::logic_error("Fail: Different size");

//             for (size_t i = 0; i < reference.size(); ++i) {
//                 if (reference[i] != infer_result[i])
//                     throw std::logic_error("Fail: Bbox different");
//             }

//         }
//     }
// }

void compareResults(const std::map<size_t, std::vector<std::vector<DetObject>>> &frame_results) {
    for (const auto &frame : frame_results) {
        size_t image_id = frame.first;
        std::cout << std::to_string(image_id) + " frame results comparision" << std::endl;
        std::vector<DetObject> reference = frame.second[0];
        for (const auto &infer_result : frame.second) {
            if (reference.size() != infer_result.size())
                throw std::runtime_error("Fail: Different number of the bboxes: " + std::to_string(reference.size()) +
                                            " vs " + std::to_string(infer_result.size()));

            for (size_t i = 0; i < reference.size(); ++i) {
                if (reference[i] != infer_result[i])
                    throw std::runtime_error("Fail: Bboxes are different");
            }

        }
    }
}

// std::map<size_t, std::vector<std::vector<DetObject>>> inferFrames(const std::vector<cv::Mat> &images, const std::string &model_path, size_t num_req, size_t iteration_count) {
//     IEWrapper detector(model_path, num_req);
//     std::cout <<  "Start Inference" << std::endl;
//     for (size_t i = 0; i < iteration_count; ++i) {
//         for (size_t i = 0; i < images.size(); ++i)
//             detector.inferAsync(images[i], i);
//     }

//     std::this_thread::sleep_for(std::chrono::milliseconds(5000));

//     return detector.infer_request_results;
// }

int main(int argc, char *argv[])
{
    if (!ParseAndCheckCommandLine(argc, argv))
        return 0;

    const std::vector<std::string> input_image_paths = {
        FLAGS_test_files + "/frame_0.png",
        FLAGS_test_files + "/frame_1.png",
        FLAGS_test_files + "/frame_2.png"
    };

    // const std::string model_path = "./models/2021.4/intel/face-detection-0205/FP32/face-detection-0205.xml";

    std::vector<cv::Mat> images = {
        cv::imread(input_image_paths[0]),
        cv::imread(input_image_paths[1]),
        cv::imread(input_image_paths[2])
    };

    const size_t num_req = 5;
    const size_t iteration_count = 10;

    IEWrapper detector(FLAGS_m_det, num_req);
    std::cout <<  "Start Inference" << std::endl;
    for (size_t i = 0; i < iteration_count; ++i) {
        for (size_t i = 0; i < images.size(); ++i)
            detector.inferAsync(images[i], i);
    }

    std::cout <<  "Results comparision" << std::endl;
    compareResults(detector.infer_request_results);

    std::cout <<  "Done!" << std::endl;
    return 0;
}