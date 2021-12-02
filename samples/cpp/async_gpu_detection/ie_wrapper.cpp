#include "ie_wrapper.hpp"
#include <samples/ocv_common.hpp>

using namespace InferenceEngine;

// size_t hash_c_string(const char* p, size_t s) {
//     size_t result = 0;
//     const size_t prime = 31;
//     for (size_t i = 0; i < s; ++i) {
//         result = p[i] + (result * prime);
//     }
//     return result;
// }

// size_t
// hash_c_string1(const char *str, size_t s)
// {
//     size_t hash = 5381;

//     for (size_t i = 0; i < s; ++i) {
//         hash = ((hash << 5) + hash) + str[i]; /* hash * 33 + c */
//     }

//     return hash;
// }

IEWrapper::IEWrapper(const std::string& network_path, size_t numRequests) : ie(), network(ie.ReadNetwork(network_path)), numRequests(numRequests), curIteration(0) {
    loadNetwork();

    for (auto i = 0; i < numRequests; i++)
    {
        infer_requests.push_back(executable_network.CreateInferRequestPtr());
    }
}

void IEWrapper::loadNetwork() {
    setInputInfo();
    // setOutputInfo();

    std::map<std::string, std::string> inference_config;
    inference_config["GPU_THROUGHPUT_STREAMS"] = "GPU_THROUGHPUT_AUTO";
    executable_network = ie.LoadNetwork(network, "GPU", inference_config);
}

void IEWrapper::inferAsync(const cv::Mat &image, size_t image_id) {
    Blob::Ptr imgBlob = wrapMat2Blob(image);

    std::unique_lock<std::mutex> lock(mutex);
    const std::string input_name = network.getInputsInfo().begin()->first;
    while (infer_requests.empty())
    {
        condition_variable.wait(lock);
    }

    auto infer_request = infer_requests.front();
    infer_requests.pop_front();

    infer_request->SetBlob(input_name, imgBlob);
    auto completion_callback = [this, image_id, infer_request](InferenceEngine::InferRequest ireq, InferenceEngine::StatusCode code)
    {
        mutex.lock();
        parseOutputBlob(infer_request, image_id);
        infer_requests.push_back(infer_request);
        mutex.unlock();
        condition_variable.notify_one();
    };


    infer_request->SetCompletionCallback<std::function<void(InferenceEngine::InferRequest, InferenceEngine::StatusCode)>>(completion_callback);
    infer_request->StartAsync();
}

void IEWrapper::parseOutputBlob(const InferenceEngine::InferRequest::Ptr infer_request, size_t image_id) {
    // const std::string output_name = network.getOutputsInfo().begin()->first;
    const std::string output_name = "boxes";
    Blob::Ptr output = infer_request->GetBlob(output_name);
    const float *data = static_cast<const float *>(output->buffer());
    auto dims = output->getTensorDesc().getDims();

    size_t dims_size = dims.size();
    size_t max_proposal_count = dims[0];
    size_t object_size = dims[dims_size - 1];
    std::vector<DetObject> det_objects;

    for (size_t i = 0; i < max_proposal_count; ++i) {

        float confidence = data[i * object_size + 4];

        if (confidence < 0.5f) {
            continue;
        }

        float bbox_x = data[i * object_size + 0];
        float bbox_y = data[i * object_size + 1];
        float bbox_w = data[i * object_size + 2] - bbox_x;
        float bbox_h = data[i * object_size + 3] - bbox_y;

        DetObject bbox = {bbox_x, bbox_y, bbox_w, bbox_h, confidence};
        det_objects.push_back(bbox);
    }

    infer_request_results[image_id].push_back(det_objects);
}

// float * IEWrapper::getOutputBlob(const std::string &output_layer_name) {
//     if (OK != infer_request.Wait(IInferRequest::WaitMode::RESULT_READY))
//         throw std::runtime_error("Waiting for inference results error");

//     Blob::Ptr output = infer_request.GetBlob(output_layer_name);
//     return static_cast<float *>(output->buffer());
// }

void IEWrapper::setInputInfo() {
    if (network.getInputsInfo().empty())
        throw std::logic_error("Network inputs info is empty");

    if (network.getInputsInfo().size() != 1)
            throw std::logic_error("Sample supports topologies with 1 input only");

    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
}

void IEWrapper::setOutputInfo() {
    if (network.getOutputsInfo().empty())
        throw std::logic_error("Network outputs info is empty");

    DataPtr output_info = network.getOutputsInfo().begin()->second;
    output_info->setLayout(Layout::NCHW);
    output_info->setPrecision(Precision::FP32);
}