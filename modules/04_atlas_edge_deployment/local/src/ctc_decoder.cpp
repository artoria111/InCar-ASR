#include "ctc_decoder.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

namespace car_asr {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

float LogAdd(float left, float right) {
    if (left == kNegativeInfinity) return right;
    if (right == kNegativeInfinity) return left;
    const float maximum = std::max(left, right);
    return maximum + std::log(std::exp(left - maximum) + std::exp(right - maximum));
}

float TotalScore(const std::pair<float, float>& scores) {
    return LogAdd(scores.first, scores.second);
}

}  // namespace

bool CTCDecoder::Init(const std::string& token_path, const Config& cfg) {
    cfg_ = cfg;
    id2token_.clear();
    token2id_.clear();

    std::ifstream file(token_path);
    if (!file.is_open()) {
        fprintf(stderr, "[CTCDecoder] Cannot open token file: %s\n",
                token_path.c_str());
        return false;
    }

    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        id2token_[id] = line;
        token2id_[line] = id;
        id++;
    }
    file.close();

    fprintf(stdout, "[CTCDecoder] Loaded %d tokens from %s\n",
            id, token_path.c_str());
    return true;
}

std::vector<int> CTCDecoder::GreedySearch(const float* logits, int T, int V) {
    std::vector<int> tokens;
    if (!logits || T <= 0 || V <= 0 || cfg_.blank_id < 0 ||
        cfg_.blank_id >= V) {
        return tokens;
    }
    int prev_token = cfg_.blank_id;

    for (int t = 0; t < T; t++) {
        // argmax
        int best = cfg_.blank_id;
        float best_score = logits[t * V];
        for (int v = 1; v < V; v++) {
            if (logits[t * V + v] > best_score) {
                best_score = logits[t * V + v];
                best = v;
            }
        }

        // CTC merge: skip blank and consecutive duplicates
        if (best != cfg_.blank_id && best != prev_token) {
            tokens.push_back(best);
        }
        prev_token = best;
    }

    return tokens;
}

std::string CTCDecoder::TokenIdsToText(const std::vector<int>& ids) {
    std::string result;
    for (int id : ids) {
        auto it = id2token_.find(id);
        if (it != id2token_.end()) {
            const std::string& token = it->second;
            if (token != "<blank>" && token != "<unk>" && token != "<sos>" && token != "<eos>") {
                result += token;
            }
        }
    }
    return result;
}

std::string CTCDecoder::GreedyDecode(const float* logits, int T, int V) {
    std::vector<int> token_ids = GreedySearch(logits, T, V);
    return TokenIdsToText(token_ids);
}

std::string CTCDecoder::BeamDecode(const float* logits, int T, int V, int beam_size) {
    if (beam_size <= 1) {
        return GreedyDecode(logits, T, V);
    }
    if (!logits || T <= 0 || V <= 0 || cfg_.blank_id < 0 ||
        cfg_.blank_id >= V) {
        return "";
    }

    // Prefix beam search. Each prefix stores log P(prefix ending in blank)
    // and log P(prefix ending in a non-blank token).
    using Prefix = std::vector<int>;
    using Scores = std::pair<float, float>;
    std::map<Prefix, Scores> beams;
    beams[{}] = {0.0f, kNegativeInfinity};

    for (int t = 0; t < T; ++t) {
        const float* frame = logits + t * V;
        const float maximum = *std::max_element(frame, frame + V);
        float normalizer = 0.0f;
        for (int token = 0; token < V; ++token) {
            normalizer += std::exp(frame[token] - maximum);
        }
        const float log_normalizer = maximum + std::log(normalizer);
        std::vector<int> candidate_tokens;
        candidate_tokens.reserve(std::max(0, V - 1));
        for (int token = 0; token < V; ++token) {
            if (token != cfg_.blank_id) candidate_tokens.push_back(token);
        }
        const size_t candidate_count = std::min(
            candidate_tokens.size(), static_cast<size_t>(beam_size * 2));
        std::partial_sort(
            candidate_tokens.begin(),
            candidate_tokens.begin() + candidate_count,
            candidate_tokens.end(),
            [frame](int left, int right) {
                return frame[left] > frame[right];
            });
        candidate_tokens.resize(candidate_count);

        std::map<Prefix, Scores> next;
        for (const auto& beam : beams) {
            const Prefix& prefix = beam.first;
            const float blank_score = beam.second.first;
            const float nonblank_score = beam.second.second;
            const float total_score = LogAdd(blank_score, nonblank_score);

            auto unchanged_insert = next.emplace(
                prefix, Scores{kNegativeInfinity, kNegativeInfinity});
            Scores& unchanged = unchanged_insert.first->second;
            const float blank_log_prob =
                frame[cfg_.blank_id] - log_normalizer;
            unchanged.first = LogAdd(
                unchanged.first, total_score + blank_log_prob);

            for (int token : candidate_tokens) {
                const float token_log_prob = frame[token] - log_normalizer;
                if (cfg_.beam_threshold > 0.0f &&
                    token_log_prob < -cfg_.beam_threshold) {
                    continue;
                }
                if (!prefix.empty() && prefix.back() == token) {
                    unchanged.second = LogAdd(
                        unchanged.second, nonblank_score + token_log_prob);
                    Prefix extended = prefix;
                    extended.push_back(token);
                    auto inserted = next.emplace(
                        extended, Scores{kNegativeInfinity, kNegativeInfinity});
                    inserted.first->second.second = LogAdd(
                        inserted.first->second.second,
                        blank_score + token_log_prob);
                } else {
                    Prefix extended = prefix;
                    extended.push_back(token);
                    auto inserted = next.emplace(
                        extended, Scores{kNegativeInfinity, kNegativeInfinity});
                    inserted.first->second.second = LogAdd(
                        inserted.first->second.second,
                        total_score + token_log_prob);
                }
            }
        }

        std::vector<std::pair<Prefix, Scores>> ranked(next.begin(), next.end());
        std::partial_sort(
            ranked.begin(),
            ranked.begin() + std::min(static_cast<size_t>(beam_size), ranked.size()),
            ranked.end(),
            [](const auto& left, const auto& right) {
                return TotalScore(left.second) > TotalScore(right.second);
            });
        beams.clear();
        const size_t keep = std::min(static_cast<size_t>(beam_size), ranked.size());
        for (size_t index = 0; index < keep; ++index) {
            beams.emplace(std::move(ranked[index]));
        }
    }

    if (beams.empty()) return "";
    const auto best = std::max_element(
        beams.begin(), beams.end(),
        [](const auto& left, const auto& right) {
            return TotalScore(left.second) < TotalScore(right.second);
        });
    return TokenIdsToText(best->first);
}

} // namespace car_asr
