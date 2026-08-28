#!/usr/bin/env python3
"""
车载噪声数据增强管线 — 动态加噪 / 变速扰动 / SpecAugment / 混响模拟

Usage:
  # 单文件增强
  python augment_audio.py --input clean.wav --noise engine_idle.wav --snr 10 --output augmented.wav

  # 批量增强（配合预处理管线）
  python augment_audio.py --batch --input_dir data/processed/train --noise_dir data/raw --output_dir data/augmented
"""

import argparse, os, sys, random, json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, convolve
from scipy.io import wavfile

# ============================================================
# 1. SNR 在线加噪
# ============================================================
def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    按指定 SNR 混合干净语音和噪声。
    SNR = 10 * log10(P_signal / P_noise)
    """
    clean = np.asarray(clean, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)

    # 对齐长度
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    start = random.randint(0, len(noise) - len(clean))
    noise = noise[start:start + len(clean)]

    # 计算能量
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-10)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)

    # 目标噪声能量: P_noise = P_signal / 10^(SNR/10)
    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (desired_noise_rms / (noise_rms + 1e-10))

    mixed = clean + noise
    # 防止削波
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * 0.99 / peak

    return mixed.astype(np.float32)


# ============================================================
# 2. 变速扰动
# ============================================================
def speed_perturb(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """
    通过重采样实现变速（0.9x ~ 1.1x），不改变采样率。
    factor=0.9: 变慢 10%; factor=1.1: 变快 10%
    """
    audio = np.asarray(audio, dtype=np.float32)
    if abs(factor - 1.0) < 0.01:
        return audio

    # Resample to new rate, then back to original
    new_len = int(len(audio) / factor)
    perturbed = resample_poly(audio, len(audio), new_len)
    # Trim or pad to match original length
    if len(perturbed) > len(audio):
        perturbed = perturbed[:len(audio)]
    elif len(perturbed) < len(audio):
        perturbed = np.pad(perturbed, (0, len(audio) - len(perturbed)))

    return perturbed.astype(np.float32)


# ============================================================
# 3. SpecAugment 频谱增强
# ============================================================
def specaugment(mel_spectrogram: np.ndarray,
                freq_mask_width: int = 15,
                time_mask_width: int = 20,
                num_freq_masks: int = 2,
                num_time_masks: int = 2) -> np.ndarray:
    """
    SpecAugment: 在 Mel 频谱上随机掩蔽频率轴和时间轴。
    输入: [time, freq] Mel 频谱
    """
    mel = mel_spectrogram.copy()
    T, F = mel.shape

    # 频率掩蔽
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_width)
        f0 = random.randint(0, F - f) if F > f else 0
        mel[:, f0:f0 + f] = mel.mean()

    # 时间掩蔽
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_width)
        t0 = random.randint(0, T - t) if T > t else 0
        mel[t0:t0 + t, :] = mel.mean()

    return mel


def apply_specaugment_to_audio(audio: np.ndarray, sample_rate: int,
                                n_mels: int = 80) -> np.ndarray:
    """对原始音频应用 SpecAugment（需经过 Mel 域变换）"""
    audio = np.asarray(audio, dtype=np.float32)

    # 简易STFT → Mel → SpecAugment → 逆STFT
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, n_fft=512, hop_length=160
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_aug = specaugment(mel_db)
    mel_power = librosa.db_to_power(mel_aug)

    # Griffin-Lim 重建
    audio_recon = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sample_rate, n_fft=512, hop_length=160
    )
    # Pad/trim
    if len(audio_recon) < len(audio):
        audio_recon = np.pad(audio_recon, (0, len(audio) - len(audio_recon)))
    else:
        audio_recon = audio_recon[:len(audio)]

    return audio_recon.astype(np.float32)


# ============================================================
# 4. 房间混响模拟
# ============================================================
def add_reverb(audio: np.ndarray, sample_rate: int,
               rt60: float = 0.5, wet_dry_ratio: float = 0.3) -> np.ndarray:
    """
    模拟简单房间混响。
    rt60: 混响时间（秒），车载环境通常 0.3-0.6s
    wet_dry_ratio: 湿信号比例
    """
    audio = np.asarray(audio, dtype=np.float32)

    # 指数衰减冲激响应
    ir_len = int(rt60 * sample_rate)
    t = np.arange(ir_len) / sample_rate
    ir = np.exp(-6.9 * t / rt60) * np.random.randn(ir_len) * 0.1
    ir = ir / (np.max(np.abs(ir)) + 1e-10)

    # 卷积
    wet = convolve(audio, ir, mode='full')[:len(audio)]
    wet = wet / (np.max(np.abs(wet)) + 1e-10) * np.max(np.abs(audio))

    return ((1 - wet_dry_ratio) * audio + wet_dry_ratio * wet).astype(np.float32)


# ============================================================
# 5. 综合增强流水线
# ============================================================
class AudioAugmentor:
    """车载语音数据增强器"""

    def __init__(self, noise_dir: str = None, sample_rate: int = 16000):
        self.sr = sample_rate
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.noise_cache = {} if noise_dir else None

    def _load_noise(self, category: str = None) -> np.ndarray:
        """随机加载一个噪声文件"""
        if not self.noise_dir:
            return np.zeros(1000, dtype=np.float32)

        noise_files = list(self.noise_dir.rglob("*.wav")) + list(self.noise_dir.rglob("*.flac"))
        if category:
            noise_files = [f for f in noise_files if category in str(f)]

        if not noise_files:
            return np.zeros(1000, dtype=np.float32)

        path = random.choice(noise_files)
        if path not in self.noise_cache:
            noise, sr = sf.read(path, dtype='float32')
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            if sr != self.sr:
                g = np.gcd(sr, self.sr)
                noise = resample_poly(noise, self.sr // g, sr // g)
            self.noise_cache[path] = noise
        return self.noise_cache[path]

    def augment(self, audio: np.ndarray, mode: str = "all") -> dict:
        """
        执行数据增强，返回增强后的音频和元数据。

        mode: "noise" | "speed" | "specaug" | "reverb" | "all"
        """
        meta = {"augmentations": []}

        if mode in ("noise", "all") and self.noise_dir:
            noise = self._load_noise()
            snr = random.uniform(5, 20)
            audio = mix_at_snr(audio, noise, snr)
            meta["augmentations"].append(f"noise_snr_{snr:.0f}dB")

        if mode in ("speed", "all"):
            factor = random.choice([0.9, 0.95, 1.0, 1.05, 1.1])
            if factor != 1.0:
                audio = speed_perturb(audio, self.sr, factor)
                meta["augmentations"].append(f"speed_{factor:.2f}x")

        if mode in ("specaug", "all"):
            try:
                audio = apply_specaugment_to_audio(audio, self.sr)
                meta["augmentations"].append("specaugment")
            except ImportError:
                pass

        if mode in ("reverb", "all"):
            rt60 = random.uniform(0.2, 0.6)
            ratio = random.uniform(0.1, 0.4)
            audio = add_reverb(audio, self.sr, rt60, ratio)
            meta["augmentations"].append(f"reverb_rt60_{rt60:.1f}s")

        return audio.astype(np.float32), meta


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="车载语音数据增强")
    p.add_argument("--input", help="输入干净语音 WAV")
    p.add_argument("--output", default="augmented.wav", help="输出路径")
    p.add_argument("--noise", help="噪声 WAV 文件")
    p.add_argument("--snr", type=float, default=10, help="信噪比 (dB)")
    p.add_argument("--speed", type=float, default=1.0, help="变速因子 (0.9-1.1)")
    p.add_argument("--reverb", action="store_true", help="添加混响")
    p.add_argument("--specaug", action="store_true", help="SpecAugment")
    p.add_argument("--batch", action="store_true", help="批量模式")
    p.add_argument("--input_dir", help="批量: 输入目录")
    p.add_argument("--noise_dir", help="批量: 噪声目录")
    p.add_argument("--output_dir", help="批量: 输出目录")
    p.add_argument("--num_augments", type=int, default=3, help="每样本增强倍数")
    return p.parse_args()


def main():
    args = parse_args()

    if args.batch:
        # 批量模式
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        augmentor = AudioAugmentor(noise_dir=args.noise_dir)
        wav_files = list(input_dir.rglob("*.wav"))

        print(f"批量增强: {len(wav_files)} 文件 × {args.num_augments} 倍")
        total = 0
        for wav_path in wav_files:
            audio, sr = sf.read(wav_path, dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # 保留原始文件
            rel_path = wav_path.relative_to(input_dir)
            orig_out = output_dir / "original" / rel_path
            orig_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(orig_out, audio, sr)

            # 生成增强变体
            for i in range(args.num_augments):
                aug_audio, meta = augmentor.augment(audio, mode="all")
                aug_out = output_dir / f"aug_{i}" / rel_path
                aug_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(aug_out, aug_audio, sr)
                total += 1

            if total % 50 == 0:
                print(f"  ... {total} 增强样本已生成")

        print(f"完成: {total} 增强样本 → {output_dir}")
    else:
        # 单文件模式
        audio, sr = sf.read(args.input, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 加噪
        if args.noise and os.path.exists(args.noise):
            noise, _ = sf.read(args.noise, dtype='float32')
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            audio = mix_at_snr(audio, noise, args.snr)
            print(f"加噪完成 (SNR={args.snr}dB)")

        # 变速
        if args.speed != 1.0:
            audio = speed_perturb(audio, sr, args.speed)
            print(f"变速完成 ({args.speed}x)")

        # SpecAugment
        if args.specaug:
            try:
                audio = apply_specaugment_to_audio(audio, sr)
                print("SpecAugment 完成")
            except ImportError:
                print("SpecAugment 跳过 (需安装 librosa)")

        # 混响
        if args.reverb:
            audio = add_reverb(audio, sr)
            print("混响完成")

        sf.write(args.output, audio, sr)
        print(f"增强完成: {args.output}")


if __name__ == "__main__":
    main()
