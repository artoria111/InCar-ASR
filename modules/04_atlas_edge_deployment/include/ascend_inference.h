#ifndef CAR_ASR_ASCEND_INFERENCE_H
#define CAR_ASR_ASCEND_INFERENCE_H

#include "common.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace car_asr {

/**
 * @brief AscendCL 推理管线封装
 *
 * 管理：ACL初始化 → Context创建 → OM模型加载 → H2D拷贝 → NPU执行 → D2H拷贝
 *
 * 使用方式：
 *   1. Init(model_path)     — 加载OM模型
 *   2. Infer(fbank_feat)    — NPU推理，返回logits
 *   3. Destroy()            — 释放资源
 */
class AscendInference {
public:
    struct Config {
        int   device_id      = 0;
        bool  enable_fusion  = true;     // 开启算子融合
        bool  enable_profiling = false;
    };

    /**
     * @brief 输入输出Tensor描述
     */
    struct TensorDesc {
        std::vector<int64_t> shape;       // e.g. {1, N, 80} for FBank input
        size_t               elem_size;   // bytes per element
        std::string          name;
    };

    struct Result {
        std::vector<float>  logits;       // CTC logits [T, vocab_size]
        int                 vocab_size;
        int                 time_steps;
        ErrorCode           error;
    };

    AscendInference() = default;
    virtual ~AscendInference() = default;
    AscendInference(const AscendInference&) = delete;
    AscendInference& operator=(const AscendInference&) = delete;

    static std::unique_ptr<AscendInference> Create(const Config& cfg);

    /**
     * @brief 初始化ACL并加载OM模型
     * @param model_path  .om离线模型文件路径
     * @return true=成功
     */
    virtual bool Init(const std::string& model_path) = 0;

    /**
     * @brief 执行一次NPU推理
     * @param features  FBank特征数据 [T * 80] 按行存储
     * @param num_frames 帧数 T
     * @return 推理结果（logits）
     */
    virtual Result Infer(const float* features, int num_frames) = 0;

    /**
     * @brief 获取模型输入Tensor描述
     */
    virtual TensorDesc GetInputDesc() const = 0;

    /**
     * @brief 获取模型输出Tensor描述
     */
    virtual TensorDesc GetOutputDesc() const = 0;

    /**
     * @brief 释放资源
     */
    virtual void Destroy() = 0;
};

} // namespace car_asr

#endif // CAR_ASR_ASCEND_INFERENCE_H
