#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

static const char help_message[] = "Print a usage message.";
static const char image_message_original_image[] = "Required. Path to 'test_files' folder.";
static const char det_model_message[] = "Required. Path to an .xml file with a x5 trained detection model.";

DEFINE_bool(h, false, help_message);
DEFINE_string(test_files, "", image_message_original_image);
DEFINE_string(m_det, "", det_model_message);

static void showUsage() {
    std::cout << std::endl;
    std::cout << "async gpu detection [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -test_files \"<path>\"               " << image_message_original_image << std::endl;
    std::cout << "    -m_det \"<path>\"               " << det_model_message << std::endl;
}