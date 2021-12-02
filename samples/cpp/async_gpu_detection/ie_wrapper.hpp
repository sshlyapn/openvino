#pragma once

#include <string>
#include <opencv2/opencv.hpp>

#include <condition_variable>
#include <mutex>
#include <ie_compound_blob.h>
#include <inference_engine.hpp>
#include <cldnn/cldnn_config.hpp>
#include <thread>
#include <list>


struct DetObject {
    double x;
    double y;
    double w;
    double h;

    double confidence;
};

inline bool operator==(const DetObject& lhs, const DetObject& rhs) {
    return (lhs.x == rhs.x) && (lhs.y == lhs.y) && (rhs.w == lhs.w) && (rhs.h == lhs.h) && (rhs.confidence == lhs.confidence);
}

inline bool operator!=(const DetObject& lhs, const DetObject& rhs) {
    return !(lhs == rhs);
}

class IEWrapper{
    private:
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork network;
        std::list<InferenceEngine::InferRequest::Ptr> infer_requests;

        std::condition_variable condition_variable;
        std::mutex mutex;

        size_t numRequests;
        size_t curIteration;
        std::condition_variable condVar;
        size_t _height;
        size_t _width;

        void loadNetwork();
        void setInputInfo();
        void setOutputInfo();
        void setUpInferenceRequest();

    public:
        InferenceEngine::ExecutableNetwork executable_network;

        std::map<size_t, std::vector<std::vector<DetObject>>> infer_request_results;

        IEWrapper(const std::string& network_path, size_t numRequests);

        void inferAsync(const cv::Mat &image, size_t image_id);
        void parseOutputBlob(const InferenceEngine::InferRequest::Ptr infer_request, size_t image_id);
        // float *getOutputBlob(const std::string &output_layer_name);
};