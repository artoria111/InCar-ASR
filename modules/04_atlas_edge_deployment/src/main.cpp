/**
 * car-asr-cli — 车载ASR推理引擎命令行演示工具
 *
 * 用法:
 *   ./car-asr-cli --model <om_path> [--wav <wav_file>] [--tokens <tokens.txt>]
 *
 * 示例:
 *   # 识别WAV文件
 *   ./car-asr-cli --model model/paraformer_small_fp16.om \
 *                 --wav test/test_audio.wav \
 *                 --tokens model/tokens.txt
 *
 * Python microphone capture is provided by scripts/microphone_demo.py.
 */

#include "asr_engine.h"
#include "common.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <getopt.h>

using namespace car_asr;

static void PrintUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:\n");
    printf("  --model, -m <path>       OM model file path (required)\n");
    printf("  --wav, -w <path>         Input WAV file\n");
    printf("  --tokens, -t <path>      Token dictionary file\n");
    printf("  --device, -d <id>        NPU device ID (default: 0)\n");
    printf("  --vad-mode <0-3>         VAD aggressiveness (default: 2)\n");
    printf("  --help, -h                Show this help\n");
}

int main(int argc, char* argv[]) {
    // Default config
    std::string model_path;
    std::string wav_path;
    std::string token_path = "model/tokens.txt";
    int device_id   = 0;
    int vad_mode    = 2;

    // Parse arguments
    static struct option long_opts[] = {
        {"model",       required_argument, 0, 'm'},
        {"wav",         required_argument, 0, 'w'},
        {"tokens",      required_argument, 0, 't'},
        {"device",      required_argument, 0, 'd'},
        {"vad-mode",    required_argument, 0, 0},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:w:t:d:h", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': model_path  = optarg; break;
            case 'w': wav_path    = optarg; break;
            case 't': token_path  = optarg; break;
            case 'd': device_id   = atoi(optarg); break;
            case 'h':
            default:  PrintUsage(argv[0]); return opt == 'h' ? 0 : 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        PrintUsage(argv[0]);
        return 1;
    }

    fprintf(stdout, "========================================\n");
    fprintf(stdout, "  Car-ASR CLI — 车载语音识别引擎\n");
    fprintf(stdout, "  Model:  %s\n", model_path.c_str());
    fprintf(stdout, "  Device: NPU %d\n", device_id);
    fprintf(stdout, "========================================\n\n");

    // 创建引擎
    ASREngine::Config cfg;
    cfg.device_id = device_id;
    cfg.vad_mode  = vad_mode;
    cfg.token_path = token_path;
    cfg.ctc_collapse_repeats = false;

    auto engine = ASREngine::Create(cfg);
    if (!engine) {
        fprintf(stderr, "Failed to create ASR engine\n");
        return 1;
    }

    // 初始化
    if (!engine->Init(model_path)) {
        fprintf(stderr, "Failed to init ASR engine\n");
        return 1;
    }

    // 识别
    if (!wav_path.empty()) {
        int sample_rate = 0;
        auto pcm = ReadWavFile(wav_path, &sample_rate);
        if (pcm.empty()) {
            fprintf(stderr,
                    "Failed to read WAV file (requires mono PCM16)\n");
            return 1;
        }
        if (sample_rate != kSampleRate) {
            fprintf(stderr,
                    "Unsupported sample rate %d; resample to %d Hz first\n",
                    sample_rate, kSampleRate);
            return 1;
        }

        fprintf(stdout, "Input audio: %.2f seconds\n",
                pcm.size() / 16000.0);

        std::string text = engine->Recognize(pcm);
        fprintf(stdout, "\n=== Recognition Result ===\n");
        fprintf(stdout, "%s\n", text.c_str());
        fprintf(stdout, "==========================\n");

        // 打印性能指标
        auto metrics = engine->GetMetrics();
        metrics.Print();

    } else {
        fprintf(stdout, "No input specified. Use --wav.\n");
    }

    fprintf(stdout, "\nDone.\n");
    return 0;
}
