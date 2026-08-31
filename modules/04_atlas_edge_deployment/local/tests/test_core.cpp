#include "audio_preprocess.h"
#include "ctc_decoder.h"
#include "vad_detector.h"
#include "wav_io.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main() {
    using namespace car_asr;
    constexpr double kPi = 3.14159265358979323846;
    bool ok = true;

    AudioPreprocessor preprocessor;
    std::vector<float> features;
    ok &= Check(preprocessor.ExtractFBank({}, features) == 0,
                "empty audio must not access pcm[0]");
    std::vector<int16_t> tone(kSampleRate);
    for (size_t index = 0; index < tone.size(); ++index) {
        tone[index] = static_cast<int16_t>(
            5000.0 * std::sin(2.0 * kPi * 440.0 * index / kSampleRate));
    }
    const int frame_count = preprocessor.ExtractFBank(tone, features);
    ok &= Check(frame_count == 98, "one second must create 98 frames");
    ok &= Check(features.size() == static_cast<size_t>(frame_count * kFbankDim),
                "FBank output dimensions");
    for (float value : features) {
        ok &= Check(std::isfinite(value), "FBank values must be finite");
        if (!ok) break;
    }

    VADDetector detector;
    VADDetector::Config vad_config;
    vad_config.start_frames = 2;
    vad_config.end_frames = 3;
    ok &= Check(detector.Init(vad_config), "VAD config");
    std::vector<int16_t> vad_audio(10 * 320, 0);
    vad_audio.insert(vad_audio.end(), tone.begin(), tone.begin() + 20 * 320);
    vad_audio.insert(vad_audio.end(), 5 * 320, 0);
    std::vector<VADDetector::SpeechSegment> segments;
    ok &= Check(detector.Detect(vad_audio, segments) == 1, "VAD one segment");
    detector.Reset();
    segments.clear();
    ok &= Check(detector.Detect(vad_audio, segments) == 1,
                "VAD reset must not leak noise state");

    const auto temporary = std::filesystem::temp_directory_path();
    const auto token_path = temporary / "incar_asr_core_tokens.txt";
    {
        std::ofstream tokens(token_path);
        tokens << "<blank>\n打\n开\n";
    }
    CTCDecoder decoder;
    CTCDecoder::Config decoder_config;
    ok &= Check(decoder.Init(token_path.string(), decoder_config), "token load");
    const std::vector<float> logits = {
        5, 0, 0,
        0, 5, 0,
        0, 5, 0,
        5, 0, 0,
        0, 0, 5,
    };
    ok &= Check(decoder.GreedyDecode(logits.data(), 5, 3) == "打开",
                "CTC greedy collapse");
    ok &= Check(decoder.BeamDecode(logits.data(), 5, 3, 3) == "打开",
                "CTC prefix beam");

    const auto wav_path = temporary / "incar_asr_core.wav";
    ok &= Check(WriteWavFile(wav_path.string(), tone, kSampleRate), "WAV write");
    int sample_rate = 0;
    const auto loaded = ReadWavFile(wav_path.string(), &sample_rate);
    ok &= Check(sample_rate == kSampleRate && loaded == tone, "WAV round trip");
    std::filesystem::remove(token_path);
    std::filesystem::remove(wav_path);

    if (!ok) return 1;
    std::cout << "All portable C++ core tests passed." << std::endl;
    return 0;
}
