import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.busi import BUSIDataset
from models.detr import DETR
from utils.loss import DETRLoss
from utils.box_ops import compute_iou
from tqdm import tqdm
import configs.config as cfg


# ─── one training epoch ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_det_loss = 0.0
    total_seg_loss = 0.0

    loop = tqdm(loader, desc="  train", leave=False)
    for batch in loop:
        images = batch[0].to(device)
        masks = batch[1].to(device)
        bboxes = batch[2].to(device)
        labels = batch[3].to(device)

        logits, boxes, seg_mask = model(images)
        
        # Create targets tuple: (image, mask, bbox, label) for each sample
        targets = [(images[i], masks[i], bboxes[i], labels[i]) for i in range(len(images))]
        
        loss, det_loss, seg_loss = criterion.loss(logits, boxes, targets, seg_mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_det_loss += det_loss.item()
        total_seg_loss += seg_loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), total_det_loss / len(loader), total_seg_loss / len(loader)


# ─── validation epoch ─────────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_det_loss = 0.0
    total_seg_loss = 0.0
    total_iou  = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            masks = batch[1].to(device)
            bboxes = batch[2].to(device)
            labels = batch[3].to(device)

            logits, boxes, seg_mask = model(images)
            
            targets = [(images[i], masks[i], bboxes[i], labels[i]) for i in range(len(images))]
            loss, det_loss, seg_loss = criterion.loss(logits, boxes, targets, seg_mask)
            total_loss += loss.item()
            total_det_loss += det_loss.item()
            total_seg_loss += seg_loss.item()

            for i in range(len(images)):
                probs    = torch.softmax(logits[i], dim=-1)
                scores   = probs[:, :2].max(dim=1).values
                best_idx = scores.argmax()

                pred_box = boxes[i][best_idx].cpu()
                gt_box   = bboxes[i].cpu()
                total_iou += compute_iou(pred_box, gt_box)
                n += 1

    return total_loss / len(loader), total_det_loss / len(loader), total_seg_loss / len(loader), total_iou / max(n, 1)


# ─── main ─────────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── datasets ──────────────────────────────────────────────────────────────
    train_set = BUSIDataset(split="train")
    val_set   = BUSIDataset(split="val")
    print(f"Train: {len(train_set)}  |  Val: {len(val_set)}\n")

    train_loader = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── model / loss / optimizer ───────────────────────────────────────────────
    model     = DETR(num_classes=cfg.NUM_CLASSES, num_queries=cfg.NUM_QUERIES, use_segmentation=True).to(device)
    criterion = DETRLoss(use_segmentation=True)

    # Backbone (pre-trained ResNet layers + projections) gets 10x lower LR than
    # the transformer + heads which are trained from scratch.
    backbone_names = {"layer1", "layer2", "layer3", "layer4",
                      "input_proj1", "input_proj2", "input_proj3", "input_proj4",
                      "fusion_conv"}
    backbone_params = [p for n, p in model.named_parameters()
                       if n.split(".")[0] in backbone_names]
    head_params     = [p for n, p in model.named_parameters()
                       if n.split(".")[0] not in backbone_names]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.LR_BACKBONE},
            {"params": head_params,     "lr": cfg.LR},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    # ── logging ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(cfg.SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.LOG_PATH),  exist_ok=True)

    log_file = open(cfg.LOG_PATH, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_det_loss", "train_seg_loss", "val_loss", "val_det_loss", "val_seg_loss", "val_miou"])

    best_val_loss = float("inf")

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Det Loss':>12}  {'Seg Loss':>12}  {'Val Loss':>10}  {'Val mIoU':>10}")
    print("-" * 82)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_det_loss, train_seg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_det_loss, val_seg_loss, val_miou = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_det_loss:>12.4f}  {train_seg_loss:>12.4f}  {val_loss:>10.4f}  {val_miou:>10.4f}")
        writer.writerow([epoch, f"{train_loss:.6f}", f"{train_det_loss:.6f}", f"{train_seg_loss:.6f}", f"{val_loss:.6f}", f"{val_det_loss:.6f}", f"{val_seg_loss:.6f}", f"{val_miou:.6f}"])
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.SAVE_PATH)

    log_file.close()
    print(f"\nBest val loss: {best_val_loss:.4f}  →  saved to {cfg.SAVE_PATH}")


if __name__ == "__main__":
    train()
