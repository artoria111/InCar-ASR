#!/usr/bin/env python3
"""
流式语音识别 Demo — 模拟实时麦克风流式输入

VAD 分片 → 逐段推理 → 增量输出，模拟车载实时交互体验

Usage:
  python3 streaming_demo.py --wav test.wav --chunk-ms 200
  # 将 WAV 按 200ms 块分片输入，模拟流式场景
"""
import numpy as np, torch, torchaudio, onnxruntime as ort, soundfile as sf
import sys, time, os, json, argparse
import warnings; warnings.filterwarnings("ignore")
os.environ["ORT_DISABLE_CPU_INFO"] = "1"

# ============================================================
MODEL = '/root/work/car-asr-engine/model/model.int8.onnx'
TOKENS = '/root/work/car-asr-engine/model/sherpa_tokens.txt'
CMVN = '/root/work/car-asr-engine/model/sherpa_cmvn.npz'

# ============================================================
class StreamingASR:
    def __init__(self):
        self.sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
        tokens_raw = open(TOKENS).read().strip().split('\n')
        self.id2token = {i: parts[0] for i, line in enumerate(tokens_raw) if (parts:=line.split())}
        cmvn = np.load(CMVN)
        self.shift = torch.tensor(cmvn['means'])
        self.scale = torch.tensor(cmvn['vars'])
        
        # VAD state
        self.vad_buffer = []
        self.speech_started = False
        self.silence_frames = 0
        self.vad_frame_ms = 20
        self.vad_threshold = 0
        self.noise_baseline = 0
        self.noise_samples = 0
        self.min_speech_frames = 10   # 200ms minimum speech
        self.max_speech_frames = 500  # 10s maximum speech
        self.silence_trigger = 30     # 600ms silence = end of speech
        
        # Result tracking
        self.results = []
        self.partial_text = ""
        self.total_audio_ms = 0

    def _vad_energy(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

    def process_chunk(self, chunk: np.ndarray, sample_rate: int) -> str:
        """
        处理一个音频块，返回当前累积的识别文本。
        首次调用返回空，直到检测到完整语音段。
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim > 1: chunk = chunk.mean(axis=1)
        
        # 1. VAD: split audio into 20ms frames
        frame_len = int(sample_rate * self.vad_frame_ms / 1000)
        is_speech = False
        
        for start in range(0, len(chunk) - frame_len + 1, frame_len):
            frame = chunk[start:start + frame_len]
            energy = self._vad_energy(frame)
            
            # Adaptive noise floor
            if self.noise_samples < 50:
                self.noise_baseline += energy
                self.noise_samples += 1
                if self.noise_samples == 50:
                    self.noise_baseline /= 50
                    self.vad_threshold = self.noise_baseline * 3.0
                continue
            
            is_speech = energy > self.vad_threshold
            
            if is_speech:
                self.vad_buffer.append(frame)
                self.silence_frames = 0
                self.speech_started = True
            elif self.speech_started:
                self.silence_frames += 1
                self.vad_buffer.append(frame)
                
                # Check if speech segment ended
                if self.silence_frames >= self.silence_trigger:
                    result = self._recognize_segment()
                    if result:
                        self.partial_text += result
                    # Reset
                    self.vad_buffer = []
                    self.speech_started = False
                    self.silence_frames = 0
        
        # If speech is ongoing and buffer is too long, force recognize
        if self.speech_started and len(self.vad_buffer) >= self.max_speech_frames:
            result = self._recognize_segment()
            if result:
                self.partial_text += result
            self.vad_buffer = self.vad_buffer[-50:]  # Keep last 50 frames as context
            self.speech_started = True
            self.silence_frames = 0
        
        return self.partial_text

    def _recognize_segment(self) -> str:
        """Run inference on the buffered speech segment"""
        if len(self.vad_buffer) < self.min_speech_frames:
            return ""
        
        audio = np.concatenate(self.vad_buffer)
        dur = len(audio) / 16000
        if dur < 0.5:  # Minimum 0.5s
            return ""
        
        # Frontend
        waveform = torch.tensor(audio).unsqueeze(0) * (1 << 15)
        fbank = torchaudio.compliance.kaldi.fbank(waveform, num_mel_bins=80, frame_length=25,
            frame_shift=10, dither=0.0, energy_floor=0.0, window_type='hamming',
            sample_frequency=16000, snip_edges=True)
        
        # LFR + CMVN
        lfr_list = []; t = 0
        while t + 7 <= fbank.shape[0]: lfr_list.append(fbank[t:t+7].flatten()); t += 6
        if not lfr_list: return ""
        lfr = (torch.stack(lfr_list) + self.shift) * self.scale
        lfr = lfr.numpy().astype(np.float32)
        lens = np.array([lfr.shape[0]], dtype=np.int32)
        
        # ONNX inference
        try:
            logits, tok_num = self.sess.run(None, {
                'speech': lfr[np.newaxis, :, :],
                'speech_lengths': lens
            })
            n_tok = min(int(tok_num[0]), 40)
            best = np.argmax(logits[0, :n_tok, :], axis=1)
            text = ''.join(self.id2token.get(int(tid), '') for tid in best
                          if int(tid) not in (0, 1, 2, 3) and int(tid) in self.id2token)
            
            if text.strip():
                result = f" [{text}]"
                self.results.append({"text": text, "duration_s": round(dur, 2)})
                return result
        except:
            pass
        return ""

    def flush(self) -> str:
        """Process any remaining buffered speech at end of stream"""
        if self.speech_started and len(self.vad_buffer) >= self.min_speech_frames:
            result = self._recognize_segment()
            if result:
                self.partial_text += result
        return self.partial_text

    def summary(self) -> dict:
        return {
            "results": self.results,
            "full_text": self.partial_text.strip(),
            "segments": len(self.results),
        }


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Streaming ASR Demo")
    parser.add_argument("--wav", required=True, help="Input WAV file")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Chunk size in ms")
    args = parser.parse_args()

    audio, sr = sf.read(args.wav, dtype='float32')
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly; import math
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000//g, sr//g)
        sr = 16000

    dur = len(audio) / sr
    chunk_samples = int(sr * args.chunk_ms / 1000)

    print(f"\n{'='*60}")
    print(f"  Streaming ASR Demo")
    print(f"  Audio: {args.wav} ({dur:.1f}s)")
    print(f"  Chunk: {args.chunk_ms}ms | Simulating real-time input")
    print(f"{'='*60}\n")

    engine = StreamingASR()
    t0 = time.time()
    current_text = ""

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) < chunk_samples / 2: break

        # Simulate streaming delay
        time.sleep(args.chunk_ms / 1000 * 0.3)  # 0.3x real-time for demo speed

        new_text = engine.process_chunk(chunk, sr)
        if new_text != current_text:
            print(f"  [{start/sr:5.1f}s] {new_text}")
            current_text = new_text

    # Flush remaining
    final = engine.flush()
    if final != current_text:
        print(f"  [final] {final}")

    elapsed = time.time() - t0
    summary = engine.summary()

    print(f"\n{'='*60}")
    print(f"  Results:")
    print(f"    Segments recognized: {summary['segments']}")
    print(f"    Full text: {summary['full_text']}")
    print(f"    Processing time: {elapsed:.1f}s (audio: {dur:.1f}s)")
    if summary['segments'] > 0:
        first_word_time = summary['results'][0]['duration_s']
        print(f"    First word latency: ~{first_word_time:.2f}s")
    print(f"{'='*60}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
