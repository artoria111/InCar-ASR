import numpy as np, onnxruntime as ort, json, os, sys, time
import soundfile as sf

ONNX = '/root/work/car-asr-engine/model/paraformer_tiny_ctc.onnx'
TOKENS = '/root/work/car-asr-engine/model/tokens.json'

print(f'Loading ONNX ({os.path.getsize(ONNX)/1e6:.1f}MB)...')
sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
with open(TOKENS) as f: tokens = json.load(f)
print('Ready.\n')

# Read/create audio
if len(sys.argv) > 1:
    audio, sr = sf.read(sys.argv[1], dtype='float32')
else:
    sr = 16000
    audio = 0.2 * (np.sin(2*np.pi*400*np.linspace(0,2,sr*2))).astype(np.float32)
if audio.ndim > 1: audio = audio.mean(axis=1)
if sr != 16000:
    from scipy.signal import resample_poly
    import math; g = math.gcd(sr, 16000)
    audio = resample_poly(audio, 16000//g, sr//g)
print(f'Audio: {len(audio)/16000:.1f}s')

# FBank
FRAME_LEN, FRAME_SHIFT = 400, 160
win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(FRAME_LEN)/(FRAME_LEN-1))
audio = np.asarray(audio, dtype=np.float32)
emp = np.zeros_like(audio); emp[0] = audio[0]; emp[1:] = audio[1:] - 0.97*audio[:-1]
nf = max(1, (len(emp)-FRAME_LEN)//FRAME_SHIFT+1)
frames = np.array([emp[i*FRAME_SHIFT:i*FRAME_SHIFT+FRAME_LEN]*win for i in range(nf)])
mag = np.abs(np.fft.rfft(frames, n=512, axis=1))
# Mel filter
fmin, fmax, nmel = 0, 8000, 80
hz2mel = lambda hz: 2595*np.log10(1+hz/700)
mel2hz = lambda m: 700*(10**(m/2595)-1)
mel_pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), nmel+2))
bins = np.clip(np.floor(513*mel_pts/16000).astype(int), 0, 256)
fbank_w = np.zeros((nmel, 257))
for m in range(nmel):
    for k in range(bins[m], bins[m+1]): fbank_w[m,k] = (k-bins[m])/max(1,bins[m+1]-bins[m])
    for k in range(bins[m+1], bins[m+2]): fbank_w[m,k] = (bins[m+2]-k)/max(1,bins[m+2]-bins[m+1])
fbank_feat = np.log(np.maximum(np.dot(mag[:,:257], fbank_w.T), 1e-10))
print(f'FBank: {fbank_feat.shape}')

# LFR (7 frames stack, 6 skip)
lfr_m, lfr_n = 7, 6
T = fbank_feat.shape[0]; lfr_out = []
t = 0
while t+lfr_m <= T:
    lfr_out.append(fbank_feat[t:t+lfr_m].flatten())
    t += lfr_n
if not lfr_out: lfr_out = [np.zeros(560)]
lfr = np.array(lfr_out, dtype=np.float32)
print(f'LFR:   {lfr.shape}')

# ONNX Runtime
t0 = time.time()
ctc_logits = sess.run(None, {'speech': lfr[np.newaxis,:,:]})[0]
print(f'ONNX:  {ctc_logits.shape} ({time.time()-t0:.0f}ms)')

# CTC greedy decode
best_ids = np.argmax(ctc_logits[0], axis=1)
result = []; prev = 0
for tid in best_ids:
    if tid != 0 and tid != prev: result.append(tid)
    prev = tid
text = ''.join(tokens.get(str(tid), '') for tid in result)

print(f'\n================================================')
print(f'  Result: "{text}"')
print(f'================================================')
if text.strip():
    print('  *** END-TO-END ASR PIPELINE WORKING! ***')
else:
    print('  (empty - expected for non-speech audio)')
