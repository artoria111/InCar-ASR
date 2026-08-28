#include "audio_preprocess.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace car_asr {

void AudioPreprocessor::PreEmphasis(const std::vector<int16_t>& pcm,
                                     std::vector<float>& out) {
    int n = static_cast<int>(pcm.size());
    out.resize(n);
    if (n == 0) return;
    out[0] = pcm[0] / 32768.0f;
    for (int i = 1; i < n; i++) {
        out[i] = pcm[i] / 32768.0f - 0.97f * (pcm[i-1] / 32768.0f);
    }
}

void AudioPreprocessor::FrameAndWindow(const std::vector<float>& signal,
                                        std::vector<std::vector<float>>& frames) {
    int signal_len = static_cast<int>(signal.size());
    int num_frames = std::max(0, (signal_len - kWindowSamples) / kShiftSamples + 1);
    frames.resize(num_frames);

    // 懒初始化汉明窗
    if (hamming_window_.empty()) {
        hamming_window_.resize(kWindowSamples);
        for (int i = 0; i < kWindowSamples; i++) {
            hamming_window_[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (kWindowSamples - 1));
        }
    }

    for (int f = 0; f < num_frames; f++) {
        int start = f * kShiftSamples;
        frames[f].resize(kWindowSamples);
        for (int i = 0; i < kWindowSamples; i++) {
            frames[f][i] = signal[start + i] * hamming_window_[i];
        }
    }
}

void AudioPreprocessor::ComputeFFT(const std::vector<float>& frame,
                                    std::vector<float>& magnitude) {
    const int n = kFftPoints;
    std::vector<std::complex<float>> values(n, {0.0f, 0.0f});
    const int frame_len = std::min(static_cast<int>(frame.size()), kWindowSamples);
    for (int i = 0; i < frame_len; ++i) {
        values[i] = {frame[i], 0.0f};
    }

    // Iterative radix-2 FFT. kFftPoints is fixed at 512.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (int length = 2; length <= n; length <<= 1) {
        const float angle = -2.0f * static_cast<float>(M_PI) / length;
        const std::complex<float> root(std::cos(angle), std::sin(angle));
        for (int start = 0; start < n; start += length) {
            std::complex<float> factor(1.0f, 0.0f);
            for (int offset = 0; offset < length / 2; ++offset) {
                const auto even = values[start + offset];
                const auto odd = values[start + offset + length / 2] * factor;
                values[start + offset] = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= root;
            }
        }
    }

    magnitude.resize(n / 2 + 1);
    for (int index = 0; index <= n / 2; ++index) {
        magnitude[index] = std::norm(values[index]);
    }
}

void AudioPreprocessor::BuildMelFilterbank() {
    if (mel_built_) return;

    float f_min = 0.0f;
    float f_max = kSampleRate / 2.0f;  // Nyquist
    int num_mels = kFbankDim;
    int fft_bins = kFftPoints / 2 + 1;

    // Hz to Mel
    auto hz2mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
    auto mel2hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

    float mel_min = hz2mel(f_min);
    float mel_max = hz2mel(f_max);

    // 等间距Mel点
    std::vector<float> mel_points(num_mels + 2);
    for (int i = 0; i < num_mels + 2; i++) {
        mel_points[i] = mel2hz(mel_min + (mel_max - mel_min) * i / (num_mels + 1));
    }

    // 频率 → FFT bin映射
    std::vector<int> bins(num_mels + 2);
    for (int i = 0; i < num_mels + 2; i++) {
        bins[i] = static_cast<int>(std::floor((kFftPoints + 1) * mel_points[i] / kSampleRate));
        bins[i] = std::min(bins[i], fft_bins - 1);
    }

    // 构建滤波器组
    mel_filterbank_.assign(num_mels * fft_bins, 0.0f);
    for (int m = 0; m < num_mels; m++) {
        for (int k = bins[m]; k < bins[m+1]; k++) {
            mel_filterbank_[m * fft_bins + k] =
                (k - bins[m]) / static_cast<float>(bins[m+1] - bins[m] + 1e-8);
        }
        for (int k = bins[m+1]; k < bins[m+2]; k++) {
            mel_filterbank_[m * fft_bins + k] =
                (bins[m+2] - k) / static_cast<float>(bins[m+2] - bins[m+1] + 1e-8);
        }
    }

    mel_built_ = true;
}

void AudioPreprocessor::MelFilterAndLog(const std::vector<float>& magnitude,
                                         std::vector<float>& fbank_feat) {
    BuildMelFilterbank();
    int fft_bins = kFftPoints / 2 + 1;
    int mag_len  = std::min(static_cast<int>(magnitude.size()), fft_bins);

    fbank_feat.resize(kFbankDim, 1e-10f);  // 小值避免log(0)

    for (int m = 0; m < kFbankDim; m++) {
        float sum = 0.0f;
        for (int k = 0; k < mag_len; k++) {
            sum += mel_filterbank_[m * fft_bins + k] * magnitude[k];
        }
        fbank_feat[m] = std::log(std::max(sum, 1e-10f));
    }
}

int AudioPreprocessor::ExtractFBank(const std::vector<int16_t>& pcm_data,
                                     std::vector<float>& features) {
    features.clear();
    if (pcm_data.empty()) return 0;

    // Step 1: 预加重
    std::vector<float> pre_emph;
    PreEmphasis(pcm_data, pre_emph);

    // Step 2: 分帧 + 汉明窗
    std::vector<std::vector<float>> frames;
    FrameAndWindow(pre_emph, frames);

    int num_frames = static_cast<int>(frames.size());
    if (num_frames == 0) return 0;

    // Step 3-4: 逐帧FFT + Mel滤波
    features.resize(num_frames * kFbankDim);

    for (int f = 0; f < num_frames; f++) {
        std::vector<float> mag;
        ComputeFFT(frames[f], mag);
        MelFilterAndLog(mag, frames[f]);  // 复用frames[f]存储结果
        std::memcpy(features.data() + f * kFbankDim,
                    frames[f].data(),
                    kFbankDim * sizeof(float));
    }

    return num_frames;
}

} // namespace car_asr
