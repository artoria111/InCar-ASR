#include "common.h"
#include <cstring>
#include <fstream>
#include <vector>
#include <cstdint>

namespace car_asr {

/**
 * @brief 读取WAV文件（PCM16格式）
 * @return PCM 16-bit samples, 空vector表示失败
 */
std::vector<int16_t> ReadWavFile(const std::string& path, int* sr_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};

    // RIFF header
    char riff[4], wave[4];
    file.read(riff, 4);  // "RIFF"
    file.ignore(4);       // file size
    file.read(wave, 4);  // "WAVE"

    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE") {
        return {};
    }

    // Read chunks
    int16_t bits_per_sample = 16;
    int32_t sample_rate     = 16000;
    int16_t num_channels    = 1;
    int16_t audio_format    = 0;
    bool found_format       = false;
    std::vector<char> data;

    while (file) {
        char chunk_id[4];
        file.read(chunk_id, 4);
        if (!file) break;

        int32_t chunk_size;
        file.read(reinterpret_cast<char*>(&chunk_size), 4);

        if (std::string(chunk_id, 4) == "fmt ") {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.ignore(6);  // byte_rate + block_align
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            // 跳过剩余fmt字节
            if (chunk_size > 16) file.ignore(chunk_size - 16);
            found_format = true;
        } else if (std::string(chunk_id, 4) == "data") {
            data.resize(chunk_size);
            file.read(data.data(), chunk_size);
        } else {
            file.ignore(chunk_size);
        }
    }

    if (!found_format || data.empty() || audio_format != 1 ||
        bits_per_sample != 16 || num_channels != 1) {
        return {};
    }
    if (sr_out) *sr_out = sample_rate;

    // 转换为int16_t
    int num_samples = data.size() / (bits_per_sample / 8);
    std::vector<int16_t> pcm(num_samples);
    std::memcpy(pcm.data(), data.data(), data.size());

    return pcm;
}

/**
 * @brief 写入WAV文件
 */
bool WriteWavFile(const std::string& path,
                  const std::vector<int16_t>& pcm,
                  int sample_rate) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    int32_t data_size = pcm.size() * sizeof(int16_t);
    int32_t file_size = 36 + data_size;

    // RIFF header
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;  // PCM
    int16_t num_channels  = 1;
    int32_t byte_rate     = sample_rate * num_channels * sizeof(int16_t);
    int16_t block_align   = num_channels * sizeof(int16_t);
    int16_t bits_per_sample = 16;

    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(pcm.data()), data_size);

    return true;
}

} // namespace car_asr
