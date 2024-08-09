// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <openvino/runtime/so_ptr.hpp>

#include "openvino/runtime/iremote_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Proxy remote tensor class.
 * This class wraps the original remote tensor and change the name of RemoteTensor
 */
class RemoteTensor : public ov::IRemoteTensor {
public:
    RemoteTensor(const ov::SoPtr<ov::ITensor>& ctx, const std::string& dev_name);
    RemoteTensor(ov::SoPtr<ov::ITensor>&& ctx, const std::string& dev_name);

    const AnyMap& get_properties() const override;
    const std::string& get_device_name() const override;

    void set_shape(ov::Shape shape) override;

    const ov::element::Type& get_element_type() const override;

    const ov::Shape& get_shape() const override;

    size_t get_size() const override;

    size_t get_byte_size() const override;

    const ov::Strides& get_strides() const override;

    static ov::SoPtr<ov::ITensor> get_hardware_tensor(const ov::SoPtr<ov::ITensor>& tensor, bool unwrap);

    void copy_to(const std::shared_ptr<ov::ITensor>& dst, size_t src_offset, size_t dst_offset, size_t size) const override {
        OPENVINO_THROW("[PROXY] Unimplemented copy_to() function call");
    };

    virtual void copy_from(const std::shared_ptr<ov::ITensor>& src, size_t src_offset, size_t dst_offset, size_t size) const override {
        OPENVINO_THROW("[PROXY] Unimplemented copy_from() function call");
    }
private:
    mutable std::string m_name;
    ov::SoPtr<ov::ITensor> m_tensor;
};

}  // namespace proxy
}  // namespace ov
