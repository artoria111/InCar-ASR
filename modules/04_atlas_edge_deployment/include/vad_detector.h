#ifndef CAR_ASR_VAD_DETECTOR_H
#define CAR_ASR_VAD_DETECTOR_H

#include "common.h"
#include <vector>
#include <cstdint>

namespace car_asr {

/**
 * @brief 自适应能量 VAD — 语音活动检测
 *
 * 对输入音频逐帧（20ms）进行语音/非语音分类。
 * 使用状态机合理切分语音段，避免短促噪声误触和语音截断。
 *
 * 状态机：
 *   SILENCE ──(语音帧)──▶  SPEECH_START
 *   SPEECH_START ──(连续N帧语音)──▶ SPEECH_ONGOING
 *   SPEECH_ONGOING ──(连续M帧静音)──▶ SPEECH_END ──▶ SILENCE
 */
class VADDetector {
public:
    enum class State {
        kSilence,
        kSpeechStart,
        kSpeechOngoing,
        kSpeechEnd,
    };

    struct Config {
        int aggressiveness    = 2;     // 0=least, 1=moderate, 2=aggressive, 3=very aggressive
        int start_frames      = 5;     // 连续语音帧阈值（触发开始）
        int end_frames        = 30;    // 连续静音帧阈值（触发结束）
        int frame_ms          = 20;    // 每帧时长ms（VAD固定20ms）
        int calibration_frames = 5;    // 初始噪声估计帧数
        float min_rms         = 120.0f; // 防止全静音导致阈值为0
    };

    /**
     * @brief 一段语音区间
     */
    struct SpeechSegment {
        int start_sample;    // 起始采样点索引
        int end_sample;      // 结束采样点索引
        bool is_speech;
    };

    VADDetector() = default;
    ~VADDetector() = default;

    /**
     * @brief 初始化VAD引擎
     */
    bool Init(const Config& cfg);

    /**
     * @brief 对完整PCM音频进行VAD检测，返回语音段列表
     * @param pcm    16kHz / 16-bit / mono
     * @param segments 输出：检测到的语音段
     * @return 语音段数量
     */
    int Detect(const std::vector<int16_t>& pcm,
               std::vector<SpeechSegment>& segments);

    /**
     * @brief 单帧检测（供流式使用）
     * @param frame  20ms音频帧 (320 samples at 16kHz)
     * @return 当前帧是否为语音
     */
    bool IsSpeech(const std::vector<int16_t>& frame);

    /**
     * @brief 获取当前状态
     */
    State GetState() const { return state_; }

    /**
     * @brief 重置检测器状态
     */
    void Reset();

private:
    Config cfg_;
    State  state_ = State::kSilence;

    int consecutive_speech_  = 0;
    int consecutive_silence_ = 0;
    int current_position_    = 0;    // 当前处理到的采样点位置
    int frame_samples_       = 320;
    float noise_energy_      = 0.0f;
    int noise_frames_        = 0;

};

} // namespace car_asr

#endif // CAR_ASR_VAD_DETECTOR_H
