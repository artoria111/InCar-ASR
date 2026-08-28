#ifndef CAR_ASR_WAV_IO_H
#define CAR_ASR_WAV_IO_H

#include <cstdint>
#include <string>
#include <vector>

namespace car_asr {

std::vector<int16_t> ReadWavFile(const std::string& path, int* sample_rate);

bool WriteWavFile(
    const std::string& path,
    const std::vector<int16_t>& pcm,
    int sample_rate);

}  // namespace car_asr

#endif  // CAR_ASR_WAV_IO_H
