#ifndef CAR_ASR_CTC_DECODER_H
#define CAR_ASR_CTC_DECODER_H

#include "common.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace car_asr {

/**
 * @brief CTC贪心解码器
 */
class CTCDecoder {
public:
    struct Config {
        int   blank_id       = 0;        // blank token index
        int   unk_id         = 1;        // unknown token index
        float beam_threshold = 0.0f;     // beam search阈值（仅beam>1）
        bool  use_lm         = false;    // 是否使用语言模型
    };

    CTCDecoder() = default;
    ~CTCDecoder() = default;

    bool Init(const std::string& token_path, const Config& cfg);

    std::string GreedyDecode(const float* logits, int T, int V);
    std::string BeamDecode(const float* logits, int T, int V, int beam_size);
    int VocabSize() const { return static_cast<int>(id2token_.size()); }

private:
    std::vector<int> GreedySearch(const float* logits, int T, int V);
    std::string TokenIdsToText(const std::vector<int>& ids);

    Config cfg_;
    std::unordered_map<int, std::string> id2token_;
    std::unordered_map<std::string, int> token2id_;
};

} // namespace car_asr

#endif // CAR_ASR_CTC_DECODER_H
