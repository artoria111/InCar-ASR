#ifndef CAR_ASR_COMMON_H
#define CAR_ASR_COMMON_H

#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

namespace car_asr {

// ============================================================
// Error codes
// ============================================================
enum class ErrorCode {
    kSuccess = 0,
    kAclInitFailed,
    kAclDeviceFailed,
    kAclContextFailed,
    kModelLoadFailed,
    kModelExecuteFailed,
    kMemcpyFailed,
    kInvalidAudio,
    kPreprocessFailed,
    kVadFailed,
    kDecodeFailed,
    kNotInitialized,
};

inline const char* ErrorStr(ErrorCode e) {
    switch (e) {
        case ErrorCode::kSuccess:           return "Success";
        case ErrorCode::kAclInitFailed:     return "ACL init failed";
        case ErrorCode::kAclDeviceFailed:   return "ACL set device failed";
        case ErrorCode::kAclContextFailed:  return "ACL create context failed";
        case ErrorCode::kModelLoadFailed:   return "OM model load failed";
        case ErrorCode::kModelExecuteFailed:return "Model execute failed";
        case ErrorCode::kMemcpyFailed:      return "Host/Device memcpy failed";
        case ErrorCode::kInvalidAudio:      return "Invalid audio data";
        case ErrorCode::kPreprocessFailed:  return "Audio preprocess failed";
        case ErrorCode::kVadFailed:         return "VAD failed";
        case ErrorCode::kDecodeFailed:      return "Token decode failed";
        case ErrorCode::kNotInitialized:    return "Engine not initialized";
        default:                            return "Unknown error";
    }
}

// ============================================================
// Audio constants
// ============================================================
constexpr int kSampleRate    = 16000;      // 16kHz
constexpr int kFrameLength   = 25;         // 25ms frame length
constexpr int kFrameShift    = 10;         // 10ms frame shift
constexpr int kWindowSamples = kSampleRate * kFrameLength / 1000;  // 400 samples
constexpr int kShiftSamples  = kSampleRate * kFrameShift  / 1000;  // 160 samples
constexpr int kFbankDim      = 80;         // 80-dim FBank features
constexpr int kFftPoints     = 512;        // FFT points
constexpr int kLfrM          = 7;          // stack 7 FBank frames
constexpr int kLfrN          = 6;          // advance 6 FBank frames
constexpr int kFeatureDim    = kFbankDim * kLfrM;  // 560-dim Paraformer input

// ============================================================
// Performance metrics
// ============================================================
struct PerfMetrics {
    double audio_duration_ms = 0.0;    // 输入音频总时长
    double preprocess_ms     = 0.0;    // 预处理耗时
    double vad_ms            = 0.0;    // VAD耗时
    double inference_ms      = 0.0;    // NPU推理耗时
    double decode_ms         = 0.0;    // CTC解码耗时
    double total_ms          = 0.0;    // 端到端总耗时
    double rtf               = 0.0;    // 实时率 = total_ms / audio_duration_ms

    void Print() const {
        std::cout << "=== Performance Metrics ===" << std::endl;
        std::cout << "  Audio duration:  " << audio_duration_ms << " ms" << std::endl;
        std::cout << "  Preprocess:      " << preprocess_ms << " ms" << std::endl;
        std::cout << "  VAD:             " << vad_ms << " ms" << std::endl;
        std::cout << "  NPU Inference:   " << inference_ms << " ms" << std::endl;
        std::cout << "  Token Decode:    " << decode_ms << " ms" << std::endl;
        std::cout << "  Total latency:   " << total_ms << " ms" << std::endl;
        std::cout << "  RTF:             " << rtf << std::endl;
    }
};

// ============================================================
// Timer utility
// ============================================================
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    void Reset() { start_ = std::chrono::high_resolution_clock::now(); }
    double ElapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    double ElapsedUs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(now - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

std::vector<int16_t> ReadWavFile(const std::string& path, int* sr_out = nullptr);
bool WriteWavFile(
    const std::string& path,
    const std::vector<int16_t>& pcm,
    int sample_rate = kSampleRate);

} // namespace car_asr

#endif // CAR_ASR_COMMON_H
