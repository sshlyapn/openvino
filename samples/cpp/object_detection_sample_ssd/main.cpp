// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>
#include <gflags/gflags.h>

#include <algorithm>
#include <inference_engine.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <vector>

#include "object_detection_sample_ssd.h"

using namespace InferenceEngine;

/**
 * @brief Checks input args
 * @param argc number of args
 * @param argv list of input arguments
 * @return bool status true(Success) or false(Fail)
 */
bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_m.empty()) {
        showUsage();
        throw std::logic_error("Model is required but not set. Please set -m option.");
    }

    if (FLAGS_i.empty()) {
        showUsage();
        throw std::logic_error("Input is required but not set. Please set -i option.");
    }

    return true;
}

/**
 * \brief The entry point for the Inference Engine object_detection sample
 * application \file object_detection_sample_ssd/main.cpp \example
 * object_detection_sample_ssd/main.cpp
 */
int main(int argc, char* argv[]) {
    try {
        /** This sample covers certain topology and cannot be generalized for any
         * object detection one **/
        // ------------------------------ Get Inference Engine version
        // ------------------------------------------------------
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";

        // --------------------------- Parsing and validation of input arguments
        // ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // ------------------------------ Read input
        // -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty())
            throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;
        // ------------------------------ Get Available Devices
        // ------------------------------------------------------
        slog::info << "Device info: " << slog::endl;
        slog::info << ie.GetVersions(FLAGS_d) << slog::endl;

        if (!FLAGS_l.empty()) {
            IExtensionPtr extension_ptr = std::make_shared<Extension>(FLAGS_l);
            ie.AddExtension(extension_ptr);
            slog::info << "Extension loaded: " << FLAGS_l << slog::endl;
        }

        if (!FLAGS_c.empty() && (FLAGS_d == "GPU" || FLAGS_d == "MYRIAD" || FLAGS_d == "HDDL")) {
            // Config for device plugin custom extension is loaded from an .xml
            // description
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, FLAGS_d);
            slog::info << "Config for " << FLAGS_d << " device plugin custom extension loaded: " << FLAGS_c
                       << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        slog::info << "Loading network files:" << slog::endl << FLAGS_m << slog::endl;

        /** Read network model **/
        CNNNetwork network = ie.ReadNetwork(FLAGS_m);
        // -----------------------------------------------------------------------------------------------------
        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------
        // -------------------------------- Prepare input blobs
        // --------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputsInfo(network.getInputsInfo());

        /**
         * Some networks have SSD-like output format (ending with DetectionOutput
         * layer), but having 2 inputs as Faster-RCNN: one for image and one for
         * "image info".
         *
         * Although object_datection_sample_ssd's main task is to support clean SSD,
         * it could score the networks with two inputs as well. For such networks
         * imInfoInputName will contain the "second" input name.
         */
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2)
            throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        std::string imageInputName, imInfoInputName;

        InputInfo::Ptr inputInfo = nullptr;

        SizeVector inputImageDims;
        /** Stores input image **/

        /** Iterating over all input blobs **/
        for (auto &item : inputsInfo) {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) {
                imageInputName = item.first;

                inputInfo = item.second;

                slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

                /** Creating first input blob **/
                Precision inputPrecision = Precision::U8;
                item.second->setPrecision(inputPrecision);
            } else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) {
                imInfoInputName = item.first;

                Precision inputPrecision = Precision::FP32;
                item.second->setPrecision(inputPrecision);
                if ((item.second->getTensorDesc().getDims()[1] != 3 &&
                     item.second->getTensorDesc().getDims()[1] != 6)) {
                    throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
                }
            }
        }

        if (inputInfo == nullptr) {
            inputInfo = inputsInfo.begin()->second;
        }
        // --------------------------- Prepare output blobs
        // -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputsInfo(network.getOutputsInfo());

        std::string outputName;
        DataPtr outputInfo;

        outputInfo = outputsInfo.begin()->second;
        outputName = outputInfo->getName();
        // SSD has an additional post-processing DetectionOutput layer
        // that simplifies output filtering, try to find it.
        if (auto ngraphFunction = network.getFunction()) {
            for (const auto &out : outputsInfo) {
                for (const auto &op : ngraphFunction->get_ops()) {
                    if (op->get_type_info() == ngraph::op::DetectionOutput::get_type_info_static() &&
                        op->get_friendly_name() == out.second->getName()) {
                        outputName = out.first;
                        outputInfo = out.second;
                        break;
                    }
                }
            }
        }

        if (outputInfo == nullptr) {
            throw std::logic_error("Can't find a DetectionOutput layer in the topology");
        }

        /** Set the precision of output data provided by the user, should be called
         * before load of the network to the device **/
        outputInfo->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Loading model to the device
        // ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;

        ExecutableNetwork executable_network = ie.LoadNetwork(network, FLAGS_d,
                                                              {{CONFIG_KEY(PERFORMANCE_HINT), CONFIG_VALUE(
                                                                                                      THROUGHPUT)}});
        std::vector<std::string> supported_config_keys =
                ie.GetMetric(FLAGS_d, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        slog::info << "Device: " << FLAGS_d << slog::endl;
        for (const auto &cfg : supported_config_keys) {
            try {
                slog::info << "  {" << cfg << " , " << executable_network.GetConfig(cfg).as<std::string>();
            } catch (...) {
            };
            slog::info << " }" << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create infer request
        // -------------------------------------------------
        slog::info << "Create infer request" << slog::endl;
        std::vector<InferRequest> infer_requests;
        auto nireq = executable_network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------
        /** Collect images data ptrs **/
        std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
        std::vector<size_t> imageWidths, imageHeights;
        for (auto &i : images) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> originalData(reader->getData());
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                originalImagesData.push_back(originalData);
                imagesData.push_back(data);
                imageWidths.push_back(reader->width());
                imageHeights.push_back(reader->height());
            }
        }
        if (imagesData.empty())
            throw std::logic_error("Valid input images were not found!");

        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
        if (batchSize != imagesData.size()) {
            slog::warn << "Number of images " + std::to_string(imagesData.size()) + " doesn't match batch size " +
                          std::to_string(batchSize)
                       << slog::endl;
            batchSize = std::min(batchSize, imagesData.size());
            slog::warn << "Number of images to be processed is " << std::to_string(batchSize) << slog::endl;
        }

        /** Creating input blob **/
        for (int n = 0; n < nireq; n++) {

            auto infer_request = executable_network.CreateInferRequest();
            infer_requests.push_back(infer_request);

            Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

            /** Filling input tensor with images. First b channel, then g and r channels
             * **/
            MemoryBlob::Ptr mimage = as<MemoryBlob>(imageInput);
            if (!mimage) {
                slog::err << "We expect image blob to be inherited from MemoryBlob, but "
                             "by fact we were not able "
                             "to cast imageInput to MemoryBlob"
                          << slog::endl;
                return 1;
            }
            // locked memory holder should be alive all time while access to its buffer
            // happens
            auto minputHolder = mimage->wmap();

            size_t num_channels = mimage->getTensorDesc().getDims()[1];
            size_t image_size = mimage->getTensorDesc().getDims()[3] * mimage->getTensorDesc().getDims()[2];

            unsigned char *data = minputHolder.as<unsigned char *>();

            /** Iterate over all input images limited by batch size  **/
            for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in
                         * bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid] =
                                imagesData.at(image_id).get()[pid * num_channels + ch];
                    }
                }
            }

            if (imInfoInputName != "") {
                Blob::Ptr input2 = infer_request.GetBlob(imInfoInputName);
                auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getTensorDesc().getDims()[1];

                /** Fill input tensor with values **/
                MemoryBlob::Ptr minput2 = as<MemoryBlob>(input2);
                if (!minput2) {
                    slog::err << "We expect input2 blob to be inherited from MemoryBlob, "
                                 "but by fact we were not able "
                                 "to cast input2 to MemoryBlob"
                              << slog::endl;
                    return 1;
                }
                // locked memory holder should be alive all time while access to its
                // buffer happens
                auto minput2Holder = minput2->wmap();
                float *p = minput2Holder.as<PrecisionTrait<Precision::FP32>::value_type *>();

                for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
                    p[image_id * imInfoDim + 0] =
                            static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
                    p[image_id * imInfoDim + 1] =
                            static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
                    for (size_t k = 2; k < imInfoDim; k++) {
                        p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
                    }
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 7. Do inference
        // ---------------------------------------------------------
        slog::info << "Start inference" << slog::endl;
        const auto niter =10;
        for (int j=0; j<niter; j++) {
            std::cout << std::endl << "ITERATION: " << j << std::endl << std::endl;
            for (int i = 0; i < nireq; i++) {
                infer_requests[i].StartAsync();
            }
            for (int i = 0; i < nireq; i++) {
                infer_requests[i].Wait();
            }

            // -----------------------------------------------------------------------------------------------------

            // --------------------------- Step 8. Process output
            // -------------------------------------------------------
            auto get_uchar = [] (const Blob::Ptr p) {
                MemoryBlob::CPtr moutput = as<MemoryBlob>(p);
                auto moutputHolder = moutput->rmap();
                return  moutputHolder.as<const PrecisionTrait<Precision::U8>::value_type *>();
            };
            for (auto b : outputsInfo) {
                for (int i = 0; i < nireq; i++) {
                    const Blob::Ptr output_blob = infer_requests[i].GetBlob(b.first);
                    auto det = get_uchar(output_blob);
                    for (int k = 0; k < nireq; k++) {
                        const Blob::Ptr output_blob_k = infer_requests[k].GetBlob(b.first);
                        auto det_k = get_uchar(output_blob_k);
                        auto res = memcmp(det_k, det, output_blob->byteSize());
                        slog::info << "Comparing output blob " << b.first << "  of " << i << "'s req" <<
                                   "to  " << k << " match: " << (res ?  "NO" : "yes") << slog::endl;
                    }
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception &error) {
        slog::err << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }


    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl
               << "This sample is an API example, for any performance measurements "
                  "please use the dedicated benchmark_app tool"
               << slog::endl;
    return 0;
}