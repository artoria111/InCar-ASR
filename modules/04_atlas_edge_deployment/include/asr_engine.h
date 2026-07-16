#ifndef CAR_ASR_ASR_ENGINE_H
#define CAR_ASR_ASR_ENGINE_H

#include "common.h"
#include <string>
#include <vector>
#include <memory>

namespace car_asr {

/**
 * @brief 车载ASR推理引擎统一接口
 *
 * 封装 VAD → FBank特征提取 → NPU推理 → CTC解码 的完整管线。
 * 使用方式：
 *   1. ASREngine::Create(config)   — 创建实例
 *   2. Init(model_path)             — 加载OM模型，初始化NPU
 *   3. Recognize(pcm)               — 识别整段音频
 *   4. RecognizeStream(chunk, end)  — 流式识别（可选）
 *   5. Destroy()                    — 释放资源
 */
class ASREngine {
public:
    struct Config {
        int   device_id        = 0;        // NPU设备ID
        int   vad_mode         = 2;        // WebRTC VAD aggressiveness (0-3)
        int   vad_silence_frames = 30;     // 连续静音帧数才判停
        int   beam_size        = 1;        // CTC beam size（1=贪心）
        bool  enable_profiling = false;    // 是否采集性能数据
        std::string token_path;            // 词典文件路径
    };

    ASREngine() = default;
    virtual ~ASREngine() = default;
    ASREngine(const ASREngine&) = delete;
    ASREngine& operator=(const ASREngine&) = delete;

    /**
     * @brief 创建引擎实例
     */
    static std::unique_ptr<ASREngine> Create(const Config& cfg);

    /**
     * @brief 初始化引擎：加载OM模型，配置NPU设备
     * @param model_path  OM离线模型文件路径
     * @return true=成功
     */
    virtual bool Init(const std::string& model_path) = 0;

    /**
     * @brief 整段PCM音频识别，返回文本
     * @param pcm_data  16kHz, 16-bit, 单声道 PCM 数据
     * @return 识别出的文本，空字符串表示识别失败
     */
    virtual std::string Recognize(const std::vector<int16_t>& pcm_data) = 0;

    /**
     * @brief 流式识别（逐帧输入）
     * @param pcm_chunk  音频片段 (16kHz, 16-bit, mono)
     * @param is_end     是否为最后一个片段
     * @return 当前累积的识别文本
     */
    virtual std::string RecognizeStream(
        const std::vector<int16_t>& pcm_chunk, bool is_end) = 0;

    /**
     * @brief 获取最近一次识别的性能指标
     */
    virtual PerfMetrics GetMetrics() const = 0;

    /**
     * @brief 重置流式识别状态
     */
    virtual void Reset() = 0;

    /**
     * @brief 引擎是否已初始化
     */
    virtual bool IsInitialized() const = 0;
};

} // namespace car_asr

#endif // CAR_ASR_ASR_ENGINE_H
