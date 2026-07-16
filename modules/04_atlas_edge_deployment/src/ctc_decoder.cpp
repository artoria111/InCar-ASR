#include "ctc_decoder.h"
#include <fstream>
#include <algorithm>
#include <queue>
#include <cmath>
#include <cstdio>

namespace car_asr {

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

    // Simplified beam search
    struct Beam {
        std::vector<int> tokens;
        float score = 0.0f;
    };

    std::vector<Beam> beams;
    beams.push_back({{}, 0.0f});

    for (int t = 0; t < T; t++) {
        std::vector<Beam> next_beams;

        for (auto& beam : beams) {
            // Extend with blank
            float blank_score = beam.score + logits[t * V + cfg_.blank_id];
            Beam blank_beam = beam;
            blank_beam.score = blank_score;
            next_beams.push_back(blank_beam);

            // Extend with non-blank tokens
            for (int v = 0; v < V; v++) {
                if (v == cfg_.blank_id) continue;

                float new_score = beam.score + logits[t * V + v];
                Beam new_beam = beam;
                if (!beam.tokens.empty() && beam.tokens.back() == v) {
                    new_beam.score = new_score - 1.0f;  // duplicate penalty
                }
                new_beam.tokens.push_back(v);
                new_beam.score = new_score;
                next_beams.push_back(new_beam);
            }
        }

        // Keep top beam_size
        std::sort(next_beams.begin(), next_beams.end(),
            [](const Beam& a, const Beam& b) { return a.score > b.score; });

        if (next_beams.size() > static_cast<size_t>(beam_size)) {
            next_beams.resize(beam_size);
        }
        beams = std::move(next_beams);
    }

    if (beams.empty()) return "";
    return TokenIdsToText(beams[0].tokens);
}

} // namespace car_asr
