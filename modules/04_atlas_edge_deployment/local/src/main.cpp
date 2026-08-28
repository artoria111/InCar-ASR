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
 */

#include "asr_engine.h"
#include "common.h"
#include "wav_io.h"
#include <cstdio>
#include <cstring>
#include <getopt.h>

using namespace car_asr;

static std::string EscapeJson(const std::string& value) {
    std::string output;
    output.reserve(value.size() + 8);
    for (unsigned char character : value) {
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character >= 0x20) output += static_cast<char>(character);
        }
    }
    return output;
}

static void PrintUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n", prog);
    printf("Options:\n");
    printf("  --model, -m <path>       OM model file path (required)\n");
    printf("  --wav, -w <path>         Input WAV file\n");
    printf("  --tokens, -t <path>      Token dictionary file\n");
    printf("  --device, -d <id>        NPU device ID (default: 0)\n");
    printf("  --vad-mode <0-3>         VAD aggressiveness (default: 2)\n");
    printf("  --beam-size <n>          CTC beam size (default: 1)\n");
    printf("  --json                    Print a final machine-readable JSON line\n");
    printf("  --help, -h                Show this help\n");
}

int main(int argc, char* argv[]) {
    // Default config
    std::string model_path;
    std::string wav_path;
    std::string token_path = "model/tokens.txt";
    int device_id   = 0;
    int vad_mode    = 2;
    int beam_size   = 1;
    bool json_output = false;

    // Parse arguments
    static struct option long_opts[] = {
        {"model",       required_argument, 0, 'm'},
        {"wav",         required_argument, 0, 'w'},
        {"tokens",      required_argument, 0, 't'},
        {"device",      required_argument, 0, 'd'},
        {"vad-mode",    required_argument, 0, 1000},
        {"beam-size",   required_argument, 0, 1001},
        {"json",        no_argument,       0, 1002},
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
            case 1000: vad_mode   = atoi(optarg); break;
            case 1001: beam_size  = atoi(optarg); break;
            case 1002: json_output = true; break;
            case 'h':
            default:  PrintUsage(argv[0]); return opt == 'h' ? 0 : 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: --model is required\n");
        PrintUsage(argv[0]);
        return 1;
    }
    if (vad_mode < 0 || vad_mode > 3 || beam_size < 1) {
        fprintf(stderr, "Error: --vad-mode must be 0-3 and --beam-size must be >= 1\n");
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
    cfg.beam_size = beam_size;
    cfg.token_path = token_path;

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
                    "Failed to read WAV file; require PCM16 mono at 16000 Hz\n");
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
        if (json_output) {
            fprintf(
                stdout,
                "{\"text\":\"%s\",\"inference_ms\":%.6f,"
                "\"total_ms\":%.6f,\"rtf\":%.9f,"
                "\"sample_rate\":%d,\"device_id\":%d}\n",
                EscapeJson(text).c_str(), metrics.inference_ms,
                metrics.total_ms, metrics.rtf, sample_rate, device_id);
        }

    } else {
        fprintf(stderr, "No input specified. Use --wav.\n");
        return 1;
    }

    fprintf(stdout, "\nDone.\n");
    return 0;
}
