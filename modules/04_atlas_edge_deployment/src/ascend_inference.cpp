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
        fprintf(stdout, "[AscendCL] aclInit OK\n");

        // 2. 设置设备
        ret = aclrtSetDevice(cfg_.device_id);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[AscendCL] aclrtSetDevice(%d) failed: %d\n",
                    cfg_.device_id, ret);
            return false;
        }
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
        result.error = ErrorCode::kSuccess;

        if (!initialized_) {
            result.error = ErrorCode::kNotInitialized;
            return result;
        }

        // 1. 准备输入数据：FBank特征 [1, num_frames, 80]
        size_t input_bytes = static_cast<size_t>(num_frames) * kFbankDim * sizeof(float);
        void* input_device = nullptr;
        aclError ret = aclrtMalloc(&input_device, input_bytes,
                                    ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        ret = aclrtMemcpy(input_device, input_bytes,
                          features, input_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            aclrtFree(input_device);
            result.error = ErrorCode::kMemcpyFailed;
            return result;
        }

        // 2. 创建输入 Dataset
        aclmdlDataset* input_dataset = aclmdlCreateDataset();
        aclDataBuffer* input_buf = aclCreateDataBuffer(input_device, input_bytes);
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
        aclrtResetDevice(cfg_.device_id);
        aclFinalize();
        initialized_ = false;
        fprintf(stdout, "[AscendCL] Destroyed.\n");
    }

private:
    Config      cfg_;
    std::string model_path_;
    bool        initialized_ = false;

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
