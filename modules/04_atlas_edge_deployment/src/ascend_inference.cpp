#include "ascend_inference.h"
#include "acl/acl.h"
#include <cstring>
#include <cstdio>

namespace car_asr {

// ============================================================
// AscendInferenceImpl — PIMPL for AscendCL
// ============================================================
class AscendInferenceImpl : public AscendInference {
public:
    explicit AscendInferenceImpl(const Config& cfg) : cfg_(cfg) {}

    ~AscendInferenceImpl() override { Destroy(); }

    bool Init(const std::string& model_path) override {
        model_path_ = model_path;

        // 1. ACL 初始化
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclInit failed: %d\n", ret);
            return false;
        }
        acl_initialized_ = true;
        fprintf(stdout, "[AscendCL] aclInit OK\n");

        // 2. 设置设备
        ret = aclrtSetDevice(cfg_.device_id);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclrtSetDevice(%d) failed: %d\n",
                    cfg_.device_id, ret);
            return false;
        }
        device_set_ = true;
        fprintf(stdout, "[AscendCL] SetDevice(%d) OK\n", cfg_.device_id);

        // 3. 创建上下文
        ret = aclrtCreateContext(&context_, cfg_.device_id);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclrtCreateContext failed: %d\n", ret);
            return false;
        }
        fprintf(stdout, "[AscendCL] CreateContext OK\n");

        // 4. 加载OM模型
        ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclmdlLoadFromFile(%s) failed: %d\n",
                    model_path.c_str(), ret);
            return false;
        }
        fprintf(stdout, "[AscendCL] Load model OK, model_id=%u\n", model_id_);

        // 5. 获取模型描述
        model_desc_ = aclmdlCreateDesc();
        ret = aclmdlGetDesc(model_desc_, model_id_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclmdlGetDesc failed: %d\n", ret);
            return false;
        }

        // 打印并保存模型输入/输出信息
        size_t num_inputs  = aclmdlGetNumInputs(model_desc_);
        size_t num_outputs = aclmdlGetNumOutputs(model_desc_);
        fprintf(stdout, "[AscendCL] Model: %zu inputs, %zu outputs\n",
                num_inputs, num_outputs);

        // --- 输入Tensor ---
        if (num_inputs > 0) {
            input_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
            input_desc_.elem_size = sizeof(float);

            aclmdlIODims dims;
            ret = aclmdlGetInputDims(model_desc_, 0, &dims);
            if (ret == ACL_SUCCESS) {
                input_desc_.name = dims.name;
                for (size_t i = 0; i < dims.dimCount; i++) {
                    input_desc_.shape.push_back(dims.dims[i]);
                }
            } else {
                // fallback: use name from aclmdlGetInputNameByIndex
                const char* name = aclmdlGetInputNameByIndex(model_desc_, 0);
                if (name) input_desc_.name = name;
            }

            fprintf(stdout, "[AscendCL]   Input[0] : %s, size=%zu, dims=",
                    input_desc_.name.c_str(), input_size_);
            for (auto d : input_desc_.shape) fprintf(stdout, "%lld ", (long long)d);
            fprintf(stdout, "\n");
        }

        // --- 输出Tensor ---
        if (num_outputs > 0) {
            output_size_ = aclmdlGetOutputSizeByIndex(model_desc_, 0);
            output_desc_.elem_size = sizeof(float);

            aclmdlIODims dims;
            ret = aclmdlGetOutputDims(model_desc_, 0, &dims);
            if (ret == ACL_SUCCESS) {
                output_desc_.name = dims.name;
                for (size_t i = 0; i < dims.dimCount; i++) {
                    output_desc_.shape.push_back(dims.dims[i]);
                }
            } else {
                const char* name = aclmdlGetOutputNameByIndex(model_desc_, 0);
                if (name) output_desc_.name = name;
            }

            fprintf(stdout, "[AscendCL]   Output[0]: %s, size=%zu, dims=",
                    output_desc_.name.c_str(), output_size_);
            for (auto d : output_desc_.shape) fprintf(stdout, "%lld ", (long long)d);
            fprintf(stdout, "\n");
        }
        if (num_outputs > 1) {
            output_length_size_ =
                aclmdlGetOutputSizeByIndex(model_desc_, 1);
            const char* name = aclmdlGetOutputNameByIndex(model_desc_, 1);
            fprintf(stdout,
                    "[AscendCL]   Output[1]: %s, size=%zu (token length)\n",
                    name ? name : "<unnamed>", output_length_size_);
        }

        if (num_inputs != 1 || num_outputs < 1 || num_outputs > 2 ||
            input_size_ == 0 || output_size_ == 0) {
            fprintf(stderr,
                    "[AscendCL] Expected one input and one or two outputs "
                    "(logits[, token_length])\n");
            return false;
        }

        // Allocate model I/O once. Reusing these buffers avoids per-request
        // device allocations and makes latency measurements meaningful.
        ret = aclrtMalloc(&input_device_, input_size_,
                          ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] Input allocation failed: %d\n", ret);
            return false;
        }
        ret = aclrtMalloc(&output_device_, output_size_,
                          ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] Output allocation failed: %d\n", ret);
            return false;
        }
        if (output_length_size_ > 0) {
            ret = aclrtMalloc(&output_length_device_, output_length_size_,
                              ACL_MEM_MALLOC_NORMAL_ONLY);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr,
                        "[AscendCL] Token-length allocation failed: %d\n", ret);
                return false;
            }
        }

        input_dataset_ = aclmdlCreateDataset();
        output_dataset_ = aclmdlCreateDataset();
        input_buffer_ = aclCreateDataBuffer(input_device_, input_size_);
        output_buffer_ = aclCreateDataBuffer(output_device_, output_size_);
        if (!input_dataset_ || !output_dataset_ ||
            !input_buffer_ || !output_buffer_) {
            fprintf(stderr, "[AscendCL] Dataset creation failed\n");
            return false;
        }
        ret = aclmdlAddDatasetBuffer(input_dataset_, input_buffer_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] Add input buffer failed: %d\n", ret);
            return false;
        }
        ret = aclmdlAddDatasetBuffer(output_dataset_, output_buffer_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] Add output buffer failed: %d\n", ret);
            return false;
        }
        if (output_length_device_) {
            output_length_buffer_ = aclCreateDataBuffer(
                output_length_device_, output_length_size_);
            if (!output_length_buffer_) {
                fprintf(stderr,
                        "[AscendCL] Token-length data buffer creation failed\n");
                return false;
            }
            ret = aclmdlAddDatasetBuffer(
                output_dataset_, output_length_buffer_);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr,
                        "[AscendCL] Add token-length output failed: %d\n", ret);
                return false;
            }
            output_length_host_.resize(output_length_size_);
        }
        output_host_.resize(output_size_ / sizeof(float));

        initialized_ = true;
        fprintf(stdout, "[AscendCL] Init complete.\n");
        return true;
    }

    Result Infer(const float* features, int num_frames) override {
        Result result;
        result.error = ErrorCode::kSuccess;

        if (!initialized_) {
            result.error = ErrorCode::kNotInitialized;
            return result;
        }

        // 1. 准备输入数据：FBank特征 [1, num_frames, 80]
        size_t input_bytes =
            static_cast<size_t>(num_frames) * kFeatureDim * sizeof(float);
        if (!features || num_frames <= 0 || input_bytes > input_size_) {
            fprintf(stderr,
                    "[AscendCL] Invalid input: frames=%d bytes=%zu capacity=%zu\n",
                    num_frames, input_bytes, input_size_);
            result.error = ErrorCode::kInvalidAudio;
            return result;
        }

        aclError ret = aclrtMemset(input_device_, input_size_, 0, input_size_);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }
        ret = aclrtMemcpy(input_device_, input_size_,
                          features, input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        // Execute with the preallocated datasets.
        ret = aclmdlExecute(model_id_, input_dataset_, output_dataset_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclmdlExecute failed: %d\n", ret);
            result.error = ErrorCode::kModelExecuteFailed;
            return result;
        }

        // Copy output to the reusable host buffer.
        ret = aclrtMemcpy(output_host_.data(), output_size_,
                          output_device_, output_size_,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
        } else {
            result.logits = output_host_;
            // 输出 shape [1, T, vocab_size] → 取 T 和 V
            if (output_desc_.shape.size() >= 3) {
                result.time_steps =
                    static_cast<int>(output_desc_.shape[output_desc_.shape.size() - 2]);
                result.vocab_size =
                    static_cast<int>(output_desc_.shape.back());
            } else if (output_desc_.shape.size() == 2) {
                result.time_steps = static_cast<int>(output_desc_.shape[0]);
                result.vocab_size = static_cast<int>(output_desc_.shape[1]);
            }
            if (result.time_steps <= 0 || result.vocab_size <= 0 ||
                static_cast<size_t>(result.time_steps) *
                    static_cast<size_t>(result.vocab_size) >
                    result.logits.size()) {
                fprintf(stderr,
                        "[AscendCL] Unsupported dynamic output shape; export a "
                        "fixed [1,T,V] logits model\n");
                result.error = ErrorCode::kModelExecuteFailed;
            }
            if (result.error == ErrorCode::kSuccess &&
                output_length_device_ && !output_length_host_.empty()) {
                ret = aclrtMemcpy(
                    output_length_host_.data(), output_length_size_,
                    output_length_device_, output_length_size_,
                    ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    result.error = ErrorCode::kMemcpyFailed;
                } else {
                    int64_t token_count = 0;
                    if (output_length_size_ >= sizeof(int64_t)) {
                        std::memcpy(
                            &token_count, output_length_host_.data(),
                            sizeof(int64_t));
                    } else if (output_length_size_ >= sizeof(int32_t)) {
                        int32_t token_count_32 = 0;
                        std::memcpy(
                            &token_count_32, output_length_host_.data(),
                            sizeof(int32_t));
                        token_count = token_count_32;
                    }
                    if (token_count > 0 &&
                        token_count <= result.time_steps) {
                        result.time_steps =
                            static_cast<int>(token_count);
                        result.logits.resize(
                            static_cast<size_t>(result.time_steps) *
                            static_cast<size_t>(result.vocab_size));
                    } else {
                        fprintf(stderr,
                                "[AscendCL] Invalid token length: %lld\n",
                                static_cast<long long>(token_count));
                        result.error = ErrorCode::kModelExecuteFailed;
                    }
                }
            }
        }

        return result;
    }

    TensorDesc GetInputDesc()  const override { return input_desc_; }
    TensorDesc GetOutputDesc() const override { return output_desc_; }

    void Destroy() override {
        if (input_dataset_) {
            aclmdlDestroyDataset(input_dataset_);
            input_dataset_ = nullptr;
        }
        if (output_dataset_) {
            aclmdlDestroyDataset(output_dataset_);
            output_dataset_ = nullptr;
        }
        if (input_buffer_) {
            aclDestroyDataBuffer(input_buffer_);
            input_buffer_ = nullptr;
        }
        if (output_buffer_) {
            aclDestroyDataBuffer(output_buffer_);
            output_buffer_ = nullptr;
        }
        if (output_length_buffer_) {
            aclDestroyDataBuffer(output_length_buffer_);
            output_length_buffer_ = nullptr;
        }
        if (input_device_) {
            aclrtFree(input_device_);
            input_device_ = nullptr;
        }
        if (output_device_) {
            aclrtFree(output_device_);
            output_device_ = nullptr;
        }
        if (output_length_device_) {
            aclrtFree(output_length_device_);
            output_length_device_ = nullptr;
        }
        if (model_desc_) {
            aclmdlDestroyDesc(model_desc_);
            model_desc_ = nullptr;
        }
        if (model_id_ != 0xFFFFFFFF) {
            aclmdlUnload(model_id_);
            model_id_ = 0xFFFFFFFF;
        }
        if (context_) {
            aclrtDestroyContext(context_);
            context_ = nullptr;
        }
        if (device_set_) {
            aclrtResetDevice(cfg_.device_id);
            device_set_ = false;
        }
        if (acl_initialized_) {
            aclFinalize();
            acl_initialized_ = false;
        }
        initialized_ = false;
        output_host_.clear();
        output_length_host_.clear();
    }

private:
    Config      cfg_;
    std::string model_path_;
    bool        initialized_ = false;
    bool        acl_initialized_ = false;
    bool        device_set_ = false;

    uint32_t     model_id_   = 0xFFFFFFFF;
    aclrtContext context_    = nullptr;
    aclmdlDesc*  model_desc_ = nullptr;

    size_t     input_size_  = 0;
    size_t     output_size_ = 0;
    size_t     output_length_size_ = 0;
    TensorDesc input_desc_;
    TensorDesc output_desc_;
    void* input_device_ = nullptr;
    void* output_device_ = nullptr;
    void* output_length_device_ = nullptr;
    aclmdlDataset* input_dataset_ = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;
    aclDataBuffer* input_buffer_ = nullptr;
    aclDataBuffer* output_buffer_ = nullptr;
    aclDataBuffer* output_length_buffer_ = nullptr;
    std::vector<float> output_host_;
    std::vector<unsigned char> output_length_host_;
};

// Factory
std::unique_ptr<AscendInference> AscendInference::Create(const Config& cfg) {
    return std::make_unique<AscendInferenceImpl>(cfg);
}

} // namespace car_asr
