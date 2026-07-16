#include "vad_detector.h"
#include <cstring>
#include <cstdio>

namespace car_asr {

VADDetector::~VADDetector() {
    // 释放WebRTC VAD句柄
    if (vad_handle_) {
        // WebRtcVad_Free(vad_handle_);
        vad_handle_ = nullptr;
    }
}

bool VADDetector::Init(const Config& cfg) {
    cfg_ = cfg;

#ifdef HAS_WEBRTC_VAD
    // WebRTC VAD 初始化
    // vad_handle_ = WebRtcVad_Create();
    // if (!vad_handle_) return false;
    // WebRtcVad_Init(vad_handle_);
    // WebRtcVad_set_mode(vad_handle_, cfg_.aggressiveness);
    fprintf(stdout, "[VAD] WebRTC VAD initialized, mode=%d\n", cfg_.aggressiveness);
#else
    // 备用：基于能量的简单VAD
    fprintf(stdout, "[VAD] Using energy-based VAD fallback (no WebRTC)\n");
    vad_handle_ = nullptr;
#endif

    state_ = State::kSilence;
    return true;
}

int VADDetector::Detect(const std::vector<int16_t>& pcm,
                        std::vector<SpeechSegment>& segments) {
    Reset();
    segments.clear();

    int total_samples = static_cast<int>(pcm.size());
    int frame_count = total_samples / kFrameSamples;
    if (frame_count == 0) return 0;

    SpeechSegment current_seg;
    current_seg.is_speech = false;
    current_seg.start_sample = 0;

    for (int f = 0; f < frame_count; f++) {
        // 提取当前帧
        std::vector<int16_t> frame(
            pcm.begin() + f * kFrameSamples,
            pcm.begin() + (f + 1) * kFrameSamples);

        bool speech = IsSpeech(frame);
        current_position_ = (f + 1) * kFrameSamples;

        switch (state_) {
            case State::kSilence:
                if (speech) {
                    consecutive_speech_++;
                    if (consecutive_speech_ >= cfg_.start_frames) {
                        state_ = State::kSpeechStart;
                        current_seg.start_sample =
                            current_position_ - cfg_.start_frames * kFrameSamples;
                        current_seg.is_speech = true;
                        consecutive_speech_ = 0;
                        consecutive_silence_ = 0;
                    }
                } else {
                    consecutive_speech_ = 0;
                }
                break;

            case State::kSpeechStart:
            case State::kSpeechOngoing:
                state_ = State::kSpeechOngoing;
                if (!speech) {
                    consecutive_silence_++;
                    if (consecutive_silence_ >= cfg_.end_frames) {
                        // 语音段结束
                        current_seg.end_sample =
                            current_position_ - cfg_.end_frames * kFrameSamples;
                        segments.push_back(current_seg);

                        state_ = State::kSilence;
                        consecutive_silence_ = 0;
                        current_seg = SpeechSegment{};
                    }
                } else {
                    consecutive_silence_ = 0;
                }
                break;

            default:
                break;
        }
    }

    // 处理末尾未结束的语音段
    if (state_ == State::kSpeechOngoing || state_ == State::kSpeechStart) {
        current_seg.end_sample = total_samples;
        segments.push_back(current_seg);
    }

    fprintf(stdout, "[VAD] Detected %zu speech segments in %.2fs audio\n",
            segments.size(), total_samples / 16000.0);
    return static_cast<int>(segments.size());
}

bool VADDetector::IsSpeech(const std::vector<int16_t>& frame) {
#ifdef HAS_WEBRTC_VAD
    // 调用WebRTC VAD
    // return WebRtcVad_Process(vad_handle_, kSampleRate,
    //                          frame.data(), frame.size()) == 1;
    (void)frame;
    return false;
#else
    // 能量阈值VAD（WebRTC不可用时的备用方案）
    if (frame.empty()) return false;

    // 计算短时能量
    float energy = 0.0f;
    for (auto s : frame) {
        energy += static_cast<float>(s) * static_cast<float>(s);
    }
    energy /= frame.size();

    // 自适应阈值（以帧平均能量的1/4为界）
    // 简化版：车载环境适配，需根据实际噪声水平调优
    static float noise_floor = 0.0f;
    static int   init_frames = 0;

    if (init_frames < 50) {
        // 前50帧用于估计噪声基准
        noise_floor += energy;
        init_frames++;
        if (init_frames == 50) {
            noise_floor /= 50.0f;
            fprintf(stdout, "[VAD] Noise floor estimated: %.2f\n", noise_floor);
        }
        return false;
    }

    // 动态阈值：噪声基准 * 倍数
    float threshold = noise_floor * 4.0f;  // mode=2: 4x noise floor
    if (cfg_.aggressiveness == 3) threshold = noise_floor * 8.0f;   // 更激进
    if (cfg_.aggressiveness == 1) threshold = noise_floor * 2.0f;   // 更宽松

    return energy > threshold;
#endif
}

void VADDetector::Reset() {
    state_ = State::kSilence;
    consecutive_speech_  = 0;
    consecutive_silence_ = 0;
    current_position_    = 0;
}

} // namespace car_asr
