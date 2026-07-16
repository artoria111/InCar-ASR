#!/usr/bin/env python3
"""端到端语音识别 — Kaldi FBank + LFR + CMVN + 完整 Paraformer ONNX"""
import numpy as np, onnxruntime as ort, json, sys, time
import torch, torchaudio, soundfile as sf

MODEL = '/root/work/car-asr-engine/model/paraformer_full.onnx'
TOKENS = '/root/work/car-asr-engine/model/tokens.json'
CMVN_NPZ = '/root/work/car-asr-engine/model/cmvn.npz'
FIXED_T = 300

# Init
sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
tokens = json.load(open(TOKENS))
cmvn = np.load(CMVN_NPZ)
shift = torch.tensor(cmvn['means'])
scale = torch.tensor(cmvn['vars'])

# Load audio
wav = sys.argv[1] if len(sys.argv) > 1 else 'test/tiny_example.wav'
audio, sr = sf.read(wav, dtype='float32')
if audio.ndim > 1: audio = audio.mean(axis=1)
if sr != 16000:
    from scipy.signal import resample_poly; import math
    g = math.gcd(sr, 16000); audio = resample_poly(audio, 16000//g, sr//g)
dur = len(audio) / 16000

t_total = time.time()

# ---- Frontend ----
t0 = time.time()
waveform = torch.tensor(audio).unsqueeze(0) * (1 << 15)

fbank = torchaudio.compliance.kaldi.fbank(
    waveform, num_mel_bins=80, frame_length=25, frame_shift=10,
    dither=0.0, energy_floor=0.0, window_type='hamming',
    sample_frequency=16000, snip_edges=True)

# Pad FBank so LFR produces same count as FunASR WavFrontend
# (torchaudio kaldi_fbank produces ~3 fewer frames due to snip_edges differences)
need_fbank = ((fbank.shape[0] // 6) + 2) * 6 + 7
if fbank.shape[0] < need_fbank:
    fbank = torch.cat([fbank, torch.zeros(need_fbank - fbank.shape[0], 80)])

# LFR: stack 7 frames, skip 6
lfr_list = []; t = 0
while t + 7 <= fbank.shape[0]:
    lfr_list.append(fbank[t:t+7].flatten()); t += 6
lfr = torch.stack(lfr_list)
orig_T = lfr.shape[0]

# CMVN: (x - mean) / std via Kaldi format (shift=-mean/std, scale=1/std)
lfr = (lfr + shift) * scale

# Pad to fixed T
if orig_T < FIXED_T:
    lfr = torch.cat([lfr, torch.zeros(FIXED_T - orig_T, 560)])
else:
    lfr = lfr[:FIXED_T]
t_frontend = time.time() - t0

# ---- ONNX ----
t0 = time.time()
dec_out, tok_len = sess.run(None, {'speech': lfr.numpy()[np.newaxis, :, :]})
t_onnx = time.time() - t0

# ---- Decode ----
n_tok = int(tok_len[0]); n_tok = min(n_tok, 8)
logits = dec_out[0, :n_tok, :]
best_ids = np.argmax(logits, axis=1)
text = ''.join(tokens[tid] for tid in best_ids if tid not in (0, 2, 3) and tid < len(tokens))

t_total = time.time() - t_total

print(f'{"="*50}')
print(f'  CAR-ASR 端到端语音识别')
print(f'{"="*50}')
print(f'  文件:   {wav}')
print(f'  时长:   {dur:.1f}s | FBank: {fbank.shape[0]} 帧 | LFR: {orig_T} 帧')
print(f'  前端:   {t_frontend*1000:.0f}ms | ONNX: {t_onnx*1000:.0f}ms')
print(f'  总耗时: {t_total*1000:.0f}ms | RTF: {t_total/dur:.4f}')
print(f'{"-"*50}')
print(f'  识别结果: "{text}"')
print(f'{"="*50}')
