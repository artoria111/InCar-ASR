#include "audio_preprocess.h"
#include "ctc_decoder.h"
#include "vad_detector.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        ++failures;
    }
}

void TestFrontend() {
    car_asr::AudioPreprocessor frontend;
    std::vector<float> features;
    Check(frontend.ExtractFBank({}, features) == 0, "empty audio is rejected");

    std::vector<int16_t> audio(car_asr::kSampleRate);
    for (size_t index = 0; index < audio.size(); ++index) {
        const double phase =
            2.0 * 3.14159265358979323846 * 440.0 * index / car_asr::kSampleRate;
        audio[index] = static_cast<int16_t>(4000.0 * std::sin(phase));
    }
    const int frames = frontend.ExtractFBank(audio, features);
    Check(frames > 0, "frontend emits frames");
    Check(
        features.size() == static_cast<size_t>(frames * car_asr::kFeatureDim),
        "frontend shape is [frames, 560]");
    bool finite = true;
    for (float value : features) finite = finite && std::isfinite(value);
    Check(finite, "frontend values are finite");
}

void TestDecoder() {
    const auto token_path =
        std::filesystem::temp_directory_path() / "incar-asr-test-tokens.txt";
    {
        std::ofstream output(token_path);
        output << "<blank> 0\n<unk> 1\n你 2\n好 3\n";
    }

    car_asr::CTCDecoder decoder;
    car_asr::CTCDecoder::Config config;
    config.collapse_repeats = true;
    Check(decoder.Init(token_path.string(), config), "token table loads");
    const float logits[] = {
        0, 0, 5, 0,
        0, 0, 5, 0,
        0, 0, 0, 5,
    };
    Check(decoder.GreedyDecode(logits, 3, 4) == "你好", "CTC repeat collapse");

    config.collapse_repeats = false;
    Check(decoder.Init(token_path.string(), config), "NAR token table loads");
    Check(
        decoder.GreedyDecode(logits, 3, 4) == "你你好",
        "Paraformer NAR keeps repeated tokens");
    std::filesystem::remove(token_path);
}

void TestVad() {
    car_asr::VADDetector detector;
    car_asr::VADDetector::Config config;
    config.calibration_frames = 3;
    config.start_frames = 2;
    config.end_frames = 2;
    Check(detector.Init(config), "VAD initializes");

    const int frame_samples = car_asr::kSampleRate * config.frame_ms / 1000;
    std::vector<int16_t> audio(frame_samples * 12, 0);
    for (int frame = 3; frame < 9; ++frame) {
        for (int index = 0; index < frame_samples; ++index) {
            audio[frame * frame_samples + index] =
                (index % 2 == 0) ? 3000 : -3000;
        }
    }
    std::vector<car_asr::VADDetector::SpeechSegment> segments;
    Check(detector.Detect(audio, segments) == 1, "VAD finds one speech segment");
    Check(
        !segments.empty() && segments[0].end_sample > segments[0].start_sample,
        "VAD segment bounds are valid");

    detector.Reset();
    std::vector<int16_t> immediate_speech(frame_samples * 6, 3000);
    segments.clear();
    Check(
        detector.Detect(immediate_speech, segments) == 1,
        "VAD does not discard speech that starts immediately");
}

}  // namespace

int main() {
    TestFrontend();
    TestDecoder();
    TestVad();
    if (failures == 0) {
        std::cout << "All host tests passed\n";
        return 0;
    }
    std::cerr << failures << " host tests failed\n";
    return 1;
}
