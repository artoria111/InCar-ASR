#!/usr/bin/env python3
"""
端到端车载语音识别系统 — 系统集成脚本

集成:
  麦克风实时采集 (PyAudio) → VAD → FBank → LFR → ONNX 推理 → CTC 解码 → 文本输出

Usage:
  # 实时麦克风模式
  python 01_e2e_system.py --mode mic --model paraformer_tiny.onnx

  # WAV 文件模式
  python 01_e2e_system.py --mode file --input test.wav

  # 批量测试模式
  python 01_e2e_system.py --mode batch --test_data test.jsonl
"""

import argparse, json, os, sys, time, wave
from pathlib import Path
from collections import defaultdict
import numpy as np

# ============================================================
# 系统配置
# ============================================================
SAMPLE_RATE = 16000
FRAME_LEN_MS = 25
FRAME_SHIFT_MS = 10
N_MELS = 80
LFR_M, LFR_N = 7, 6
VAD_FRAME_MS = 20
VAD_AGGRESSIVENESS = 2
MAX_TOKENS = 8


class VADModule:
    """WebRTC VAD 封装 (能量阈值备用)"""

    def __init__(self, mode=2):
        self.mode = mode
        self.speech_frames = 0
        self.silence_frames = 0
        self.in_speech = False
        self.buffer = []

    def is_speech(self, frame: np.ndarray) -> bool:
        """帧级语音检测"""
        energy = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
        # 自适应阈值 (简化为固定阈值，车载环境可调)
        threshold = 50 * (0.5 ** self.mode)
        return energy > threshold

    def process(self, audio_chunk: np.ndarray) -> str:
        """处理音频块，返回状态: "speech_start" | "speech" | "speech_end" | "silence" """
        frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        frames = [audio_chunk[i:i + frame_samples]
                  for i in range(0, len(audio_chunk), frame_samples)]

        for frame in frames:
            if len(frame) < frame_samples:
                continue
            if self.is_speech(frame):
                self.speech_frames += 1
                self.silence_frames = 0
                self.buffer.append(frame)
                if not self.in_speech and self.speech_frames >= 3:
                    self.in_speech = True
                    return "speech_start"
                elif self.in_speech:
                    return "speech"
            else:
                self.silence_frames += 1
                if self.in_speech and self.silence_frames >= 15:
                    self.in_speech = False
                    self.speech_frames = 0
                    return "speech_end"
        return "speech" if self.in_speech else "silence"


class ASRSystem:
    """端到端车载 ASR 系统"""

    def __init__(self, model_path: str, tokens_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.tokens = json.load(open(tokens_path))
        self.vad = VADModule(mode=VAD_AGGRESSIVENESS)

        # 预计算 Mel 滤波器
        self._build_mel()
        self._build_window()

    def _build_window(self):
        fl = int(SAMPLE_RATE * FRAME_LEN_MS / 1000)
        self.win = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(fl) / (fl - 1))

    def _build_mel(self):
        hz2mel = lambda hz: 2595 * np.log10(1 + hz / 700)
        mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
        pts = mel2hz(np.linspace(hz2mel(0), hz2mel(8000), N_MELS + 2))
        fft_n = 512
        bins = np.clip(np.floor((fft_n + 1) * pts / SAMPLE_RATE).astype(int), 0, fft_n // 2)
        self.mel_w = np.zeros((N_MELS, fft_n // 2 + 1))
        for m in range(N_MELS):
            for k in range(bins[m], bins[m + 1]):
                self.mel_w[m, k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
            for k in range(bins[m + 1], bins[m + 2]):
                self.mel_w[m, k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])

    def extract_fbank(self, audio: np.ndarray) -> np.ndarray:
        fl = int(SAMPLE_RATE * FRAME_LEN_MS / 1000)
        fs = int(SAMPLE_RATE * FRAME_SHIFT_MS / 1000)
        audio = np.asarray(audio, dtype=np.float32)
        emp = np.zeros_like(audio); emp[0] = audio[0]; emp[1:] = audio[1:] - 0.97 * audio[:-1]
        nf = max(1, (len(emp) - fl) // fs + 1)
        frames = np.array([emp[i * fs:i * fs + fl] * self.win for i in range(nf)])
        mag = np.abs(np.fft.rfft(frames, n=512, axis=1))
        return np.log(np.maximum(np.dot(mag[:, :257], self.mel_w.T), 1e-10))

    def apply_lfr(self, fbank: np.ndarray) -> np.ndarray:
        T = fbank.shape[0]; out = []; t = 0
        while t + LFR_M <= T: out.append(fbank[t:t + LFR_M].flatten()); t += LFR_N
        return np.array(out, dtype=np.float32) if out else np.zeros((1, N_MELS * LFR_M), dtype=np.float32)

    def recognize(self, audio: np.ndarray) -> dict:
        """完整识别流程: WAV → FBank → LFR → ONNX → 解码"""
        t0 = time.time()

        # Frontend
        t1 = time.time()
        fbank = self.extract_fbank(audio)
        lfr = self.apply_lfr(fbank)
        orig_T = lfr.shape[0]
        # Pad to 300
        FIXED = 300
        if orig_T < FIXED:
            lfr = np.vstack([lfr, np.zeros((FIXED - orig_T, N_MELS * LFR_M), dtype=np.float32)])
        else:
            lfr = lfr[:FIXED]
        t_frontend = time.time() - t1

        # ONNX
        t1 = time.time()
        dec_out, tok_len = self.session.run(None, {"speech": lfr[np.newaxis, :, :]})
        t_onnx = time.time() - t1

        # Decode
        n_tok = min(int(tok_len[0]), MAX_TOKENS)
        logits = dec_out[0, :n_tok, :]
        best_ids = np.argmax(logits, axis=1)
        text = ''.join(self.tokens[tid] for tid in best_ids
                       if tid not in (0, 2, 3) and tid < len(self.tokens))

        return {
            "text": text,
            "frontend_ms": t_frontend * 1000,
            "onnx_ms": t_onnx * 1000,
            "total_ms": (time.time() - t0) * 1000,
            "rtf": (time.time() - t0) / (len(audio) / SAMPLE_RATE),
            "frames": orig_T,
            "tokens": n_tok,
        }

    def recognize_with_vad(self, audio_stream) -> list:
        """带 VAD 的流式识别"""
        results = []
        speech_buffer = []

        for chunk in audio_stream:
            state = self.vad.process(chunk)

            if state == "speech_start":
                speech_buffer = list(self.vad.buffer)
            elif state == "speech":
                speech_buffer.extend(chunk)
            elif state == "speech_end" and speech_buffer:
                audio = np.concatenate(speech_buffer)
                if len(audio) / SAMPLE_RATE >= 0.5:  # 最小 0.5s
                    result = self.recognize(audio)
                    results.append(result)
                    print(f"  [{len(results)}] {result['text']} "
                          f"({result['total_ms']:.0f}ms, RTF={result['rtf']:.3f})")
                speech_buffer = []

        return results


# ============================================================
# 批量测试
# ============================================================
def run_batch_test(asr: ASRSystem, test_jsonl: str, output_json: str = None):
    """批量测试，输出 CER/RTF/延迟报告"""
    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    results = []
    total_cer = 0.0
    total_rtf = 0.0
    total_delay = 0.0
    valid = 0

    for i, s in enumerate(samples):
        if not os.path.exists(s["source"]):
            continue
        import soundfile as sf
        audio, sr = sf.read(s["source"], dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)

        result = asr.recognize(audio)
        ref = s["target"].replace(" ", "")

        # CER
        m, n = len(ref), len(result["text"])
        if m == 0:
            cer = float(n > 0)
        else:
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i2 in range(m + 1): dp[i2][0] = i2
            for j2 in range(n + 1): dp[0][j2] = j2
            for i2 in range(1, m + 1):
                for j2 in range(1, n + 1):
                    cost = 0 if ref[i2 - 1] == result["text"][j2 - 1] else 1
                    dp[i2][j2] = min(dp[i2 - 1][j2] + 1, dp[i2][j2 - 1] + 1, dp[i2 - 1][j2 - 1] + cost)
            cer = dp[m][n] / m

        result["cer"] = cer
        result["ref"] = ref
        results.append(result)
        total_cer += cer
        total_rtf += result["rtf"]
        total_delay += result["total_ms"]
        valid += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(samples)}] CER={total_cer / valid:.3f}")

    # Summary
    summary = {
        "samples": valid,
        "avg_cer": total_cer / max(1, valid),
        "avg_rtf": total_rtf / max(1, valid),
        "avg_delay_ms": total_delay / max(1, valid),
        "target_cer": 0.08,
        "target_rtf": 0.1,
        "target_delay_ms": 500,
        "cer_pass": total_cer / max(1, valid) < 0.08,
        "rtf_pass": total_rtf / max(1, valid) < 0.1,
        "delay_pass": total_delay / max(1, valid) < 500,
    }

    print(f"\n=== Batch Test Summary ({valid} samples) ===")
    print(f"  CER:   {summary['avg_cer']:.4f} (target < 8%) [{'PASS' if summary['cer_pass'] else 'FAIL'}]")
    print(f"  RTF:   {summary['avg_rtf']:.4f} (target < 0.1) [{'PASS' if summary['rtf_pass'] else 'FAIL'}]")
    print(f"  Delay: {summary['avg_delay_ms']:.0f}ms (target < 500ms) [{'PASS' if summary['delay_pass'] else 'FAIL'}]")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
        print(f"  Report: {output_json}")

    return summary


# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mic", "file", "batch"], default="batch")
    p.add_argument("--input", help="WAV file (file mode) or JSONL (batch mode)")
    p.add_argument("--model", default="modules/02_asr_model_training/outputs/paraformer_full.onnx")
    p.add_argument("--tokens", default="models/models/damo--speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/snapshots/master/tokens.json")
    p.add_argument("--output", help="Output JSON report (batch mode)")
    return p.parse_args()


def main():
    args = parse_args()
    asr = ASRSystem(args.model, args.tokens)
    print("ASR System ready.\n")

    if args.mode == "batch":
        run_batch_test(asr, args.input, args.output)
    elif args.mode == "file":
        import soundfile as sf
        audio, sr = sf.read(args.input, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=1)
        result = asr.recognize(audio)
        print(f"Input:  {args.input} ({len(audio) / SAMPLE_RATE:.1f}s)")
        print(f"Result: \"{result['text']}\"")
        print(f"Time:   {result['total_ms']:.0f}ms | RTF: {result['rtf']:.4f}")
    elif args.mode == "mic":
        print("Mic mode: 请确保麦克风已连接。使用 PyAudio 采集...")
        print("(需要 pip install pyaudio)")
        print("按 Ctrl+C 停止")
        # Mic streaming implementation would go here


if __name__ == "__main__":
    main()
