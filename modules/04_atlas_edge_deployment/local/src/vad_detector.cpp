#include "vad_detector.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace car_asr {

VADDetector::~VADDetector() = default;

bool VADDetector::Init(const Config& cfg) {
    if (cfg.aggressiveness < 0 || cfg.aggressiveness > 3 ||
        cfg.start_frames <= 0 || cfg.end_frames <= 0 ||
        cfg.frame_ms != 20 || cfg.minimum_rms <= 0.0f ||
        cfg.noise_alpha < 0.0f || cfg.noise_alpha >= 1.0f) {
        fprintf(stderr, "[VAD] Invalid configuration\n");
        return false;
    }
    cfg_ = cfg;
    fprintf(stdout, "[VAD] Adaptive energy VAD initialized, mode=%d\n",
            cfg_.aggressiveness);
    Reset();
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
    if (frame.size() != kFrameSamples) return false;

    double energy = 0.0;
    for (auto s : frame) {
        const double normalized = static_cast<double>(s) / 32768.0;
        energy += normalized * normalized;
    }
    const float rms = static_cast<float>(std::sqrt(energy / frame.size()));
    static const float ratios[] = {1.8f, 2.5f, 3.5f, 5.0f};
    const float threshold = std::max(
        cfg_.minimum_rms, noise_floor_ * ratios[cfg_.aggressiveness]);
    const bool speech = rms >= threshold;
    if (!speech) {
        noise_floor_ = cfg_.noise_alpha * noise_floor_
            + (1.0f - cfg_.noise_alpha) * rms;
    }
    return speech;
}

void VADDetector::Reset() {
    state_ = State::kSilence;
    consecutive_speech_  = 0;
    consecutive_silence_ = 0;
    current_position_    = 0;
    noise_floor_ = cfg_.minimum_rms / 4.0f;
}

} // namespace car_asr
