#include "asr_engine.h"
#include "ascend_inference.h"
#include "audio_preprocess.h"
#include "vad_detector.h"
#include "ctc_decoder.h"

namespace car_asr {

class ASREngineImpl : public ASREngine {
public:
    explicit ASREngineImpl(const Config& cfg)
        : cfg_(cfg), metrics_() {}

    ~ASREngineImpl() override {
        if (ascend_) ascend_->Destroy();
    }

    bool Init(const std::string& model_path) override {
        // 1. 初始化 AscendCL 推理模块
        AscendInference::Config acl_cfg;
        acl_cfg.device_id       = cfg_.device_id;
        acl_cfg.enable_fusion   = true;
        acl_cfg.enable_profiling = cfg_.enable_profiling;

        ascend_ = AscendInference::Create(acl_cfg);
        if (!ascend_->Init(model_path)) {
            fprintf(stderr, "[ASREngine] AscendInference init failed\n");
            return false;
        }

        // 2. 初始化 VAD
        VADDetector::Config vad_cfg;
        vad_cfg.aggressiveness = cfg_.vad_mode;
        vad_cfg.end_frames     = cfg_.vad_silence_frames;

        if (!vad_.Init(vad_cfg)) {
            fprintf(stderr, "[ASREngine] VAD init failed\n");
            return false;
        }

        // 3. 初始化 CTC 解码器（如果提供了词典）
        if (cfg_.token_path.empty()) {
            fprintf(stderr, "[ASREngine] token_path is required\n");
            return false;
        }
        CTCDecoder::Config ctc_cfg;
        ctc_cfg.blank_id = 0;
        ctc_cfg.collapse_repeats = cfg_.ctc_collapse_repeats;
        if (!ctc_.Init(cfg_.token_path, ctc_cfg)) {
            fprintf(stderr, "[ASREngine] decoder init failed\n");
            return false;
        }

        initialized_ = true;
        fprintf(stdout, "[ASREngine] Initialized successfully.\n");
        return true;
    }

    std::string Recognize(const std::vector<int16_t>& pcm_data) override {
        if (!initialized_ || pcm_data.empty()) return "";

        Timer total_timer;
        metrics_ = PerfMetrics{};
        metrics_.audio_duration_ms = pcm_data.size() * 1000.0 / kSampleRate;

        // === Step 1: VAD 语音活动检测 ===
        Timer vad_timer;
        std::vector<VADDetector::SpeechSegment> segments;
        vad_.Detect(pcm_data, segments);
        metrics_.vad_ms = vad_timer.ElapsedMs();

        if (segments.empty()) {
            fprintf(stdout, "[ASREngine] VAD: no speech detected\n");
            metrics_.total_ms = total_timer.ElapsedMs();
            return "";
        }

        // 合并所有语音段
        std::vector<int16_t> speech_pcm;
        for (auto& seg : segments) {
            if (seg.is_speech && seg.end_sample > seg.start_sample) {
                speech_pcm.insert(speech_pcm.end(),
                    pcm_data.begin() + seg.start_sample,
                    pcm_data.begin() + seg.end_sample);
            }
        }

        if (speech_pcm.empty()) {
            metrics_.total_ms = total_timer.ElapsedMs();
            return "";
        }

        // === Step 2: FBank 特征提取 ===
        Timer prep_timer;
        std::vector<float> features;
        int num_frames = preprocessor_.ExtractFBank(speech_pcm, features);
        metrics_.preprocess_ms = prep_timer.ElapsedMs();
        if (num_frames <= 0 || features.empty()) {
            fprintf(stderr, "[ASREngine] FBank extraction failed\n");
            metrics_.total_ms = total_timer.ElapsedMs();
            return "";
        }

        fprintf(stdout, "[ASREngine] FBank: %d frames extracted in %.2f ms\n",
                num_frames, metrics_.preprocess_ms);

        // === Step 3: NPU 推理 ===
        Timer infer_timer;
        auto result = ascend_->Infer(features.data(), num_frames);
        metrics_.inference_ms = infer_timer.ElapsedMs();

        if (result.error != ErrorCode::kSuccess) {
            fprintf(stderr, "[ASREngine] NPU inference failed: %s\n",
                    ErrorStr(result.error));
            metrics_.total_ms = total_timer.ElapsedMs();
            return "";
        }

        fprintf(stdout, "[ASREngine] NPU inference: %.2f ms, output [%d, %d]\n",
                metrics_.inference_ms, result.time_steps, result.vocab_size);

        // === Step 4: Paraformer token decoding (optional CTC collapsing) ===
        Timer decode_timer;
        std::string text;
        if (cfg_.beam_size <= 1) {
            text = ctc_.GreedyDecode(result.logits.data(), result.time_steps, result.vocab_size);
        } else {
            text = ctc_.BeamDecode(result.logits.data(), result.time_steps,
                                    result.vocab_size, cfg_.beam_size);
        }
        metrics_.decode_ms = decode_timer.ElapsedMs();

        metrics_.total_ms = total_timer.ElapsedMs();
        if (metrics_.audio_duration_ms > 0.0) {
            metrics_.rtf = metrics_.total_ms / metrics_.audio_duration_ms;
        }

        return text;
    }

    std::string RecognizeStream(
        const std::vector<int16_t>& pcm_chunk, bool is_end) override {
        if (!initialized_) return "";
        stream_buffer_.insert(
            stream_buffer_.end(), pcm_chunk.begin(), pcm_chunk.end());
        if (!is_end) return "";
        auto buffered_audio = std::move(stream_buffer_);
        stream_buffer_.clear();
        return Recognize(buffered_audio);
    }

    PerfMetrics GetMetrics() const override { return metrics_; }
    void Reset() override {
        vad_.Reset();
        stream_buffer_.clear();
        metrics_ = PerfMetrics{};
    }
    bool IsInitialized() const override { return initialized_; }

private:
    Config      cfg_;
    bool        initialized_ = false;
    PerfMetrics metrics_;

    std::unique_ptr<AscendInference> ascend_;
    VADDetector                       vad_;
    AudioPreprocessor                 preprocessor_;
    CTCDecoder                        ctc_;
    std::vector<int16_t>              stream_buffer_;
};

// Factory
std::unique_ptr<ASREngine> ASREngine::Create(const Config& cfg) {
    return std::make_unique<ASREngineImpl>(cfg);
}

} // namespace car_asr
