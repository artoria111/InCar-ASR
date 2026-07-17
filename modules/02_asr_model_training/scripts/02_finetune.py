#!/usr/bin/env python3
"""
Paraformer-tiny 微调 — 使用预计算特征（离线 FunASR WavFrontend + CMVN）

Usage:
  uv run python modules/02_asr_model_training/scripts/02_finetune.py \
      --epochs 10 --batch_size 8 --lr 1e-4
"""

import argparse, json, os, sys, time
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

ctc_weight = 0.3

# ============================================================
# Cached Dataset
# ============================================================
class CachedDataset(Dataset):
    def __init__(self, cache_path, vocab_size):
        data = torch.load(cache_path, weights_only=False)
        self.feats = data['feats']
        self.token_ids = data['token_ids']
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.token_ids[idx]


def collate_fn(batch):
    feats_list, targets_list = zip(*batch)
    max_t = max(f.size(0) for f in feats_list)
    dim = feats_list[0].size(1)
    padded_feats = torch.zeros(len(feats_list), max_t, dim)
    feat_lens = torch.zeros(len(feats_list), dtype=torch.long)
    for i, f in enumerate(feats_list):
        padded_feats[i, :f.size(0)] = f
        feat_lens[i] = f.size(0)

    max_len = max(t.size(0) for t in targets_list)
    padded_targets = torch.zeros(len(targets_list), max_len, dtype=torch.long)
    target_lens = torch.zeros(len(targets_list), dtype=torch.long)
    for i, t in enumerate(targets_list):
        padded_targets[i, :t.size(0)] = t
        target_lens[i] = t.size(0)

    return padded_feats, feat_lens, padded_targets, target_lens


# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, optimizer, scaler, device, ctc_loss, ce_loss):
    model.train()
    total_loss = 0.0
    for feats, feat_lens, targets, target_lens in loader:
        feats = feats.to(device); feat_lens = feat_lens.to(device)
        targets = targets.to(device); target_lens = target_lens.to(device)

        optimizer.zero_grad()
        with autocast():
            encoder_out, encoder_out_lens = model.encode(feats, feat_lens)
            if isinstance(encoder_out, tuple): encoder_out = encoder_out[0]

            pred_outs = model.calc_predictor(encoder_out, encoder_out_lens)
            pre_acoustic_embeds = pred_outs[0]
            pre_token_length = pred_outs[1].round().long().clamp(min=1)

            dec_outs = model.cal_decoder_with_predictor(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length)
            decoder_out = dec_outs[0]

            # CTC loss
            ctc_out = model.ctc.ctc_lo(encoder_out)
            ctc_log = nn.functional.log_softmax(ctc_out, dim=-1)
            loss_ctc = ctc_loss(ctc_log.transpose(0, 1), targets, feat_lens, target_lens)

            # Decoder CE loss — per-sample
            loss_ce = torch.tensor(0.0, device=device)
            n_ce = 0
            for i in range(decoder_out.size(0)):
                n_pred = min(pre_token_length[i].item(), decoder_out.size(1))
                n_tgt = target_lens[i].item()
                n = min(n_pred, n_tgt)
                if n > 0:
                    loss_ce += ce_loss(decoder_out[i, :n, :], targets[i, :n])
                    n_ce += 1
            loss_ce = loss_ce / max(1, n_ce)

            loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_ce

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def validate(model, loader, device, ctc_loss, ce_loss):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for feats, feat_lens, targets, target_lens in loader:
            feats = feats.to(device); feat_lens = feat_lens.to(device)
            targets = targets.to(device); target_lens = target_lens.to(device)

            encoder_out, encoder_out_lens = model.encode(feats, feat_lens)
            if isinstance(encoder_out, tuple): encoder_out = encoder_out[0]

            pred_outs = model.calc_predictor(encoder_out, encoder_out_lens)
            pre_acoustic_embeds = pred_outs[0]
            pre_token_length = pred_outs[1].round().long().clamp(min=1)

            dec_outs = model.cal_decoder_with_predictor(
                encoder_out, encoder_out_lens, pre_acoustic_embeds, pre_token_length)
            decoder_out = dec_outs[0]

            ctc_out = model.ctc.ctc_lo(encoder_out)
            ctc_log = nn.functional.log_softmax(ctc_out, dim=-1)
            loss_ctc = ctc_loss(ctc_log.transpose(0, 1), targets, feat_lens, target_lens)

            loss_ce_v = torch.tensor(0.0, device=device)
            for i in range(decoder_out.size(0)):
                n_pred = min(pre_token_length[i].item(), decoder_out.size(1))
                n_tgt = target_lens[i].item()
                n = min(n_pred, n_tgt)
                if n > 0:
                    loss_ce_v += ce_loss(decoder_out[i, :n, :], targets[i, :n])
            loss_ce_v = loss_ce_v / max(1, decoder_out.size(0))

            total_loss += (ctc_weight * loss_ctc + (1 - ctc_weight) * loss_ce_v).item()
            n_batches += 1

    return total_loss / max(1, n_batches)


# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--output_dir', default='modules/02_asr_model_training/outputs')
    p.add_argument('--model_dir', default='models/models/damo--speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/snapshots/master')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = 'modules/02_asr_model_training/data'

    print('=' * 60)
    print(f'  Paraformer-tiny Fine-tuning  |  {device}')
    print('=' * 60)

    # Load model
    print('\n[1/3] Loading pretrained model...')
    from funasr import AutoModel
    funasr_model = AutoModel(model=args.model_dir, device='cpu', disable_update=True)
    model = funasr_model.model.to(device).train()

    # Freeze encoder — preserve pretrained feature extraction
    for param in model.encoder.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total:,} total, {trainable:,} trainable (encoder frozen)')

    # Vocab
    with open(f'{args.model_dir}/tokens.json', 'r', encoding='utf-8') as f:
        vocab_size = len(json.load(f))

    # Load cached data
    print('\n[2/3] Loading cached features...')
    train_ds = CachedDataset(f'{DATA_DIR}/train_cached.pt', vocab_size)
    val_ds = CachedDataset(f'{DATA_DIR}/val_cached.pt', vocab_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {args.batch_size}')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    os.makedirs(args.output_dir, exist_ok=True)

    # Train
    print(f'\n[3/3] Training ({args.epochs} epochs)...')
    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, ctc_loss, ce_loss)
        val_loss = validate(model, val_loader, device, ctc_loss, ce_loss)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f'  Epoch {epoch:2d}/{args.epochs} | '
              f'train={train_loss:.4f} | val={val_loss:.4f} | lr={lr:.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'train_loss': train_loss},
                       f'{args.output_dir}/best_model.pt')
            print(f'    [SAVED] best model checkpoint')

    elapsed = (time.time() - t0) / 60
    print(f'\nDone in {elapsed:.0f}min | Best val_loss: {best_val_loss:.4f}')
    print(f'Checkpoint: {args.output_dir}/best_model.pt')


if __name__ == '__main__':
    main()
