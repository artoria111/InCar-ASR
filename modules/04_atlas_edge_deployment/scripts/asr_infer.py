#!/usr/bin/env python3
"""端到端车载语音识别 — 完整 Paraformer ONNX (Encoder + Predictor + Decoder)"""
import numpy as np, onnxruntime as ort, json, os, sys, time
import soundfile as sf

MODEL = '/root/work/car-asr-engine/model/paraformer_full.onnx'
TOKENS = '/root/work/car-asr-engine/model/tokens.json'
FIXED_T, N_MELS = 300, 80
FRAME_LEN, FRAME_SHIFT = 400, 160
LFR_M, LFR_N = 7, 6

class ASREngine:
    def __init__(self):
        self.sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
        self.tokens = json.load(open(TOKENS))
        self._build_mel_filter()
        self.win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(FRAME_LEN)/(FRAME_LEN-1))

    def _build_mel_filter(self):
        hz2mel=lambda hz:2595*np.log10(1+hz/700)
        mel2hz=lambda m:700*(10**(m/2595)-1)
        pts=mel2hz(np.linspace(hz2mel(0),hz2mel(8000),N_MELS+2))
        bins=np.clip(np.floor(513*pts/16000).astype(int),0,256)
        self.mel_w=np.zeros((N_MELS,257))
        for m in range(N_MELS):
            for k in range(bins[m],bins[m+1]): self.mel_w[m,k]=(k-bins[m])/max(1,bins[m+1]-bins[m])
            for k in range(bins[m+1],bins[m+2]): self.mel_w[m,k]=(bins[m+2]-k)/max(1,bins[m+2]-bins[m+1])

    def extract_fbank(self, audio):
        audio=np.asarray(audio,dtype=np.float32)
        emp=np.zeros_like(audio); emp[0]=audio[0]; emp[1:]=audio[1:]-0.97*audio[:-1]
        nf=max(1,(len(emp)-FRAME_LEN)//FRAME_SHIFT+1)
        frames=np.array([emp[i*FRAME_SHIFT:i*FRAME_SHIFT+FRAME_LEN]*self.win for i in range(nf)])
        mag=np.abs(np.fft.rfft(frames,n=512,axis=1))
        return np.log(np.maximum(np.dot(mag[:,:257],self.mel_w.T),1e-10))

    def apply_lfr(self, fbank):
        T=fbank.shape[0]; out=[]
        t=0
        while t+LFR_M<=T: out.append(fbank[t:t+LFR_M].flatten()); t+=LFR_N
        return np.array(out,dtype=np.float32) if out else np.zeros((1,N_MELS*LFR_M),dtype=np.float32)

    def infer(self, audio_wav):
        t_total=time.time()

        # Frontend
        t0=time.time(); fbank=self.extract_fbank(audio_wav)
        t_fbank=time.time()-t0
        t0=time.time(); lfr=self.apply_lfr(fbank)
        t_lfr=time.time()-t0

        # Pad/crop to fixed T
        T=lfr.shape[0]; orig_T=T
        if T<FIXED_T:
            lfr=np.vstack([lfr,np.zeros((FIXED_T-T,N_MELS*LFR_M),dtype=np.float32)])
        else:
            lfr=lfr[:FIXED_T]; orig_T=FIXED_T

        # ONNX: Full Paraformer
        t0=time.time()
        decoder_out, token_lens = self.sess.run(None, {'speech':lfr[np.newaxis,:,:]})
        t_onnx=time.time()-t0

        # Decode (full decoder output, no CTC needed)
        n_tokens = int(token_lens[0])
        logits = decoder_out[0, :n_tokens, :]  # [N, vocab]
        best_ids = np.argmax(logits, axis=1)

        # Remove sos(2), eos(3), blank(0)
        text=''
        for tid in best_ids:
            if tid not in (0,2,3) and tid<len(self.tokens):
                text+=self.tokens[tid]

        t_total=time.time()-t_total
        return text, {
            'fbank_ms':t_fbank*1000,'lfr_ms':t_lfr*1000,'onnx_ms':t_onnx*1000,
            'total_ms':t_total*1000,'frames':T,'tokens':n_tokens
        }

def main():
    engine = ASREngine()
    print('ASR Engine (Full Paraformer) ready.\n')

    if len(sys.argv)>1:
        wav=sys.argv[1]
        audio,sr=sf.read(wav,dtype='float32')
        if audio.ndim>1: audio=audio.mean(axis=1)
        if sr!=16000:
            from scipy.signal import resample_poly
            import math; g=math.gcd(sr,16000)
            audio=resample_poly(audio,16000//g,sr//g)
    else:
        sr=16000; audio=0.2*np.sin(2*np.pi*440*np.linspace(0,2,sr*2)).astype(np.float32)
        wav='(test tone)'

    dur=len(audio)/16000
    text,stats=engine.infer(audio)

    print(f'Input:  {wav} ({dur:.1f}s)')
    print(f'FBank:  {stats["frames"]} frames ({stats["fbank_ms"]:.0f}ms)')
    print(f'ONNX:   {stats["onnx_ms"]:.0f}ms  ({stats["tokens"]} output tokens)')
    print(f'Total:  {stats["total_ms"]:.0f}ms')
    print(f'\n  Result: "{text}"')
    print(f'  *** FULL MODEL WORKING ***' if text else '  (empty)')

if __name__=='__main__':
    main()
