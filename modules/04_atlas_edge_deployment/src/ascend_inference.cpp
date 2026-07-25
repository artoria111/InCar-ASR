#include "ascend_inference.h"
#include "acl/acl.h"
#include <algorithm>
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
        if (initialized_) {
            fprintf(stderr, "[AscendCL] Init called twice\n");
            return false;
        }
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
        if (num_inputs != 1 || num_outputs != 1) {
            fprintf(
                stderr,
                "[AscendCL] This CTC runtime requires exactly one input and one output\n");
            return false;
        }
        if (aclmdlGetInputDataType(model_desc_, 0) != ACL_FLOAT ||
            aclmdlGetOutputDataType(model_desc_, 0) != ACL_FLOAT) {
            fprintf(
                stderr,
                "[AscendCL] C++ runtime currently requires float32 model I/O\n");
            return false;
        }

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

        initialized_ = true;
        fprintf(stdout, "[AscendCL] Init complete.\n");
        return true;
    }

    Result Infer(const float* features, int num_frames) override {
        Result result;

        if (!initialized_) {
            result.error = ErrorCode::kNotInitialized;
            return result;
        }
        if (!features || num_frames <= 0) {
            result.error = ErrorCode::kPreprocessFailed;
            return result;
        }

        // 1. 准备输入数据：FBank特征 [1, num_frames, 80]
        size_t input_bytes = static_cast<size_t>(num_frames) * kFbankDim * sizeof(float);
        if (input_size_ == 0 || input_size_ % sizeof(float) != 0) {
            result.error = ErrorCode::kPreprocessFailed;
            return result;
        }
        std::vector<float> input_host(input_size_ / sizeof(float), 0.0f);
        const size_t copy_bytes = std::min(input_bytes, input_size_);
        std::memcpy(input_host.data(), features, copy_bytes);
        if (input_bytes > input_size_) {
            fprintf(
                stderr,
                "[AscendCL] Warning: feature input truncated from %zu to %zu bytes\n",
                input_bytes, input_size_);
        }
        void* input_device = nullptr;
        aclError ret = aclrtMalloc(&input_device, input_size_,
                                    ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        ret = aclrtMemcpy(input_device, input_size_,
                          input_host.data(), input_size_,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(input_device);
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        // 2. 创建输入 Dataset
        aclmdlDataset* input_dataset = aclmdlCreateDataset();
        aclDataBuffer* input_buf = aclCreateDataBuffer(input_device, input_size_);
        aclmdlAddDatasetBuffer(input_dataset, input_buf);

        // 3. 创建输出 Dataset + 输出缓冲区
        void* output_device = nullptr;
        ret = aclrtMalloc(&output_device, output_size_,
                          ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            aclmdlDestroyDataset(input_dataset);
            aclDestroyDataBuffer(input_buf);
            aclrtFree(input_device);
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        aclmdlDataset* output_dataset = aclmdlCreateDataset();
        aclDataBuffer* output_buf = aclCreateDataBuffer(output_device, output_size_);
        aclmdlAddDatasetBuffer(output_dataset, output_buf);

        // 4. NPU 推理
        ret = aclmdlExecute(model_id_, input_dataset, output_dataset);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclmdlExecute failed: %d\n", ret);
            aclmdlDestroyDataset(output_dataset);
            aclDestroyDataBuffer(output_buf);
            aclrtFree(output_device);
            aclmdlDestroyDataset(input_dataset);
            aclDestroyDataBuffer(input_buf);
            aclrtFree(input_device);
            result.error = ErrorCode::kModelExecuteFailed;
            return result;
        }

        // 5. 拷贝输出到 Host
        size_t output_elems = output_size_ / sizeof(float);
        std::vector<float> output_host(output_elems);
        ret = aclrtMemcpy(output_host.data(), output_size_,
                          output_device, output_size_,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
        } else {
            result.logits = std::move(output_host);
            // 输出 shape [1, T, vocab_size] → 取 T 和 V
            if (output_desc_.shape.size() >= 3) {
                result.time_steps = static_cast<int>(output_desc_.shape[1]);
                result.vocab_size  = static_cast<int>(output_desc_.shape[2]);
            } else if (output_desc_.shape.size() == 2) {
                result.time_steps = static_cast<int>(output_desc_.shape[0]);
                result.vocab_size = static_cast<int>(output_desc_.shape[1]);
            }
            const size_t expected = static_cast<size_t>(
                std::max(result.time_steps, 0)) *
                static_cast<size_t>(std::max(result.vocab_size, 0));
            if (result.time_steps <= 0 || result.vocab_size <= 0 ||
                expected > result.logits.size()) {
                fprintf(stderr, "[AscendCL] Unsupported or invalid output shape\n");
                result.logits.clear();
                result.error = ErrorCode::kModelExecuteFailed;
            } else {
                result.error = ErrorCode::kSuccess;
            }
        }

        // 6. 清理
        aclmdlDestroyDataset(output_dataset);
        aclDestroyDataBuffer(output_buf);
        aclrtFree(output_device);
        aclmdlDestroyDataset(input_dataset);
        aclDestroyDataBuffer(input_buf);
        aclrtFree(input_device);

        return result;
    }

    TensorDesc GetInputDesc()  const override { return input_desc_; }
    TensorDesc GetOutputDesc() const override { return output_desc_; }

    void Destroy() override {
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
    TensorDesc input_desc_;
    TensorDesc output_desc_;
};

// Factory
std::unique_ptr<AscendInference> AscendInference::Create(const Config& cfg) {
    return std::make_unique<AscendInferenceImpl>(cfg);
}

} // namespace car_asr
