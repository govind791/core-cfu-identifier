from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from ml.augmentations import get_train_transform, get_val_transform
from ml.dataset import build_dataloaders
from ml.evaluate import evaluate_epoch
from ml.model import CFUDetectorModel, CFUDetectorLoss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = total_focal = total_size = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="  train", leave=False):
        img = batch["image"].to(device)
        hm_tgt = batch["heatmap"].to(device)
        sz_tgt = batch["size_map"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            hm_pred, sz_pred = model(img)
            losses = criterion(hm_pred, sz_pred, hm_tgt, sz_tgt)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += losses["total"].item()
        total_focal += losses["focal"].item()
        total_size += losses["size"].item()
        n_batches += 1
    n = max(1, n_batches)
    return {"loss": total_loss/n, "focal": total_focal/n, "size": total_size/n}


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=0,
        train_transform=get_train_transform(), val_transform=get_val_transform(),
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = CFUDetectorModel(pretrained=not args.no_pretrain).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model_state"])

    criterion = CFUDetectorLoss(size_weight=args.size_weight)
    backbone_params = list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters())
    head_params = list(model.upsample.parameters()) + list(model.neck.parameters()) + list(model.head.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        log = {
            "epoch": epoch, "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"], "val_mae": val_metrics.get("count_mae", -1),
            "lr": optimizer.param_groups[1]["lr"], "elapsed_s": time.time()-t0,
        }
        history.append(log)
        print(f"Epoch {epoch:3d}/{args.epochs} | train={log['train_loss']:.4f} | val={log['val_loss']:.4f} | mae={log['val_mae']:.2f}")
        model.save(str(output_dir/"latest.pt"), extra_info=log)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            model.save(str(output_dir/"best.pt"), extra_info=log)
            print(f"  New best saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    (output_dir/"history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/exp1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
