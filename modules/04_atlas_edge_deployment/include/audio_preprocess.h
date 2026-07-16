#ifndef CAR_ASR_AUDIO_PREPROCESS_H
#define CAR_ASR_AUDIO_PREPROCESS_H

#include "common.h"
#include <vector>
#include <cstdint>

namespace car_asr {

/**
 * @brief 音频预处理：重采样 → 预加重 → 分帧 → FBank特征提取
 *
 * 输入：PCM 16kHz 16-bit 单声道音频
 * 输出：80维 FBank 特征矩阵 [num_frames, kFbankDim]
 */
class AudioPreprocessor {
public:
    AudioPreprocessor() = default;
    ~AudioPreprocessor() = default;

    /**
     * @brief 从PCM音频提取FBank特征
     * @param pcm_data  16kHz / 16-bit / mono 原始音频
     * @param features  输出：FBank特征 [num_frames * kFbankDim]，逐帧存储
     * @return 帧数
     */
    int ExtractFBank(const std::vector<int16_t>& pcm_data,
                     std::vector<float>& features);

    /**
     * @brief 获取当前Mel滤波器组（首次提取时计算，后续复用）
     */
    const std::vector<float>& GetMelFilterbank() const { return mel_filterbank_; }

private:
    // Step 1: 预加重  y[n] = x[n] - 0.97 * x[n-1]
    void PreEmphasis(const std::vector<int16_t>& pcm,
                     std::vector<float>& out);

    // Step 2: 分帧 + 汉明窗
    void FrameAndWindow(const std::vector<float>& signal,
                        std::vector<std::vector<float>>& frames);

    // Step 3: FFT (使用FFTW3或内置KissFFT)
    void ComputeFFT(const std::vector<float>& frame,
                    std::vector<float>& magnitude);

    // Step 4: Mel滤波 + 对数压缩
    void MelFilterAndLog(const std::vector<float>& magnitude,
                         std::vector<float>& fbank_feat);

    // 懒加载Mel滤波器组
    void BuildMelFilterbank();

    // 归一化
    void Normalize(std::vector<std::vector<float>>& features);

    bool mel_built_ = false;
    std::vector<float> mel_filterbank_;  // [kFbankDim * kFftPoints/2]
    std::vector<float> hamming_window_;  // [kWindowSamples]
};

} // namespace car_asr

#endif // CAR_ASR_AUDIO_PREPROCESS_H
