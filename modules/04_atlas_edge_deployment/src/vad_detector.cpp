#include "vad_detector.h"
#include <algorithm>
#include <cstdio>

namespace car_asr {

bool VADDetector::Init(const Config& cfg) {
    cfg_ = cfg;
    if (cfg_.aggressiveness < 0 || cfg_.aggressiveness > 3 ||
        cfg_.start_frames < 1 || cfg_.end_frames < 1 ||
        cfg_.frame_ms < 10 || cfg_.frame_ms > 30) {
        fprintf(stderr, "[VAD] Invalid configuration\n");
        return false;
    }
    frame_samples_ = kSampleRate * cfg_.frame_ms / 1000;
    fprintf(stdout,
            "[VAD] Adaptive energy VAD initialized, mode=%d frame=%dms\n",
            cfg_.aggressiveness, cfg_.frame_ms);
    Reset();
    return true;
}

int VADDetector::Detect(const std::vector<int16_t>& pcm,
                        std::vector<SpeechSegment>& segments) {
    Reset();
    segments.clear();

    int total_samples = static_cast<int>(pcm.size());
    int frame_count = total_samples / frame_samples_;
    if (frame_count == 0) return 0;

    SpeechSegment current_seg;
    current_seg.is_speech = false;
    current_seg.start_sample = 0;

    for (int f = 0; f < frame_count; f++) {
        // 提取当前帧
        std::vector<int16_t> frame(
            pcm.begin() + f * frame_samples_,
            pcm.begin() + (f + 1) * frame_samples_);

        bool speech = IsSpeech(frame);
        current_position_ = (f + 1) * frame_samples_;

        switch (state_) {
            case State::kSilence:
                if (speech) {
                    consecutive_speech_++;
                    if (consecutive_speech_ >= cfg_.start_frames) {
                        state_ = State::kSpeechStart;
                        current_seg.start_sample =
                            current_position_ - cfg_.start_frames * frame_samples_;
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
                            current_position_ - cfg_.end_frames * frame_samples_;
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
    if (frame.size() != static_cast<size_t>(frame_samples_)) return false;

    // 计算短时能量
    float energy = 0.0f;
    for (auto s : frame) {
        energy += static_cast<float>(s) * static_cast<float>(s);
    }
    energy /= frame.size();

    static constexpr float kMultipliers[] = {1.8f, 2.5f, 4.0f, 6.0f};
    const float minimum_energy = cfg_.min_rms * cfg_.min_rms;

    // Estimate the noise floor first, but do not discard speech that begins
    // immediately at the start of a clip.
    if (noise_frames_ < cfg_.calibration_frames) {
        if (energy > minimum_energy * kMultipliers[cfg_.aggressiveness]) {
            return true;
        }
        noise_frames_++;
        noise_energy_ += (energy - noise_energy_) / noise_frames_;
        return false;
    }

    const float threshold =
        std::max(noise_energy_ * kMultipliers[cfg_.aggressiveness],
                 minimum_energy);
    const bool speech = energy > threshold;

    // Track the slowly-changing cabin noise only on non-speech frames.
    if (!speech) {
        noise_energy_ = 0.98f * noise_energy_ + 0.02f * energy;
    }
    return speech;
}

void VADDetector::Reset() {
    state_ = State::kSilence;
    consecutive_speech_  = 0;
    consecutive_silence_ = 0;
    current_position_    = 0;
    noise_energy_        = 0.0f;
    noise_frames_        = 0;
}

} // namespace car_asr
