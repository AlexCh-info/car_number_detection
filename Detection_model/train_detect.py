import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from pathlib import Path

from podgotovka.project_numbers.loss import DetectionLoss
from second_model import PlateDetectorGrid
from second_data import DataNumber

def calculate_iou(preds, targets):
    # cxcywh -> x1y1x2y2
    p_x1, p_y1 = preds[:, 0] - preds[:, 2]/2, preds[:, 1] - preds[:, 3]/2
    p_x2, p_y2 = preds[:, 0] + preds[:, 2]/2, preds[:, 1] + preds[:, 3]/2
    t_x1, t_y1 = targets[:, 0] - targets[:, 2]/2, targets[:, 1] - targets[:, 3]/2
    t_x2, t_y2 = targets[:, 0] + targets[:, 2]/2, targets[:, 1] + targets[:, 3]/2

    ix1, iy1 = torch.max(p_x1, t_x1), torch.max(p_y1, t_y1)
    ix2, iy2 = torch.min(p_x2, t_x2), torch.min(p_y2, t_y2)

    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    union = (preds[:, 2] * preds[:, 3]) + (targets[:, 2] * targets[:, 3]) - inter
    return (inter / (union + 1e-6)).mean().item()

def train_stage(model, dataloader,  criterion, optimizer, device, epoch):
    model.train()
    train_loss = 0
    train_iou = 0
    count = 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        mask = targets[..., 0] == 1
        if mask.any():
            train_iou += calculate_iou(output[mask][..., 1:5], targets[mask][..., 1:5])
            count += 1
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{train_loss / len(dataloader):.3f}", "lr": optimizer.param_groups[0]['lr'],
                          "IoU": f"{(train_iou / (count + 1e-6)):.3f}"})

def val_stage(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    total_iou = 0
    count = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validation split")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            mask = targets[..., 0] == 1
            if mask.any():
                total_iou += calculate_iou(output[mask][..., 1:5], targets[mask][..., 1:5])
                count += 1

            val_loss += criterion(output, targets).item()
        avg_loss = val_loss / len(dataloader)
        avg_iou = total_iou / (count + 1e-6)
        print(f"Validation loss: {avg_loss:.4f}, IoU: {avg_iou:.3f}")
        return avg_loss

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PlateDetectorGrid().to(device)
    model.load_state_dict(torch.load('../project_numbers/checkpoints/detector_ccpd_fn_best.pth', ))
    criterion = DetectionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    stages = ["ccpd_rotate", "ccpd_tilt", "ccpd_weather"]
    data_root = Path("D:/prof/podgotovka/data/CCPD2019")
    base_lr = 1e-3

    for i, stage in enumerate(stages):

        current_lr = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        print(f"Starting learning on lib {stage}")
        full_data = DataNumber((data_root / stage), grid_size=7, img_size=224)

        max_len = 30000
        if len(full_data) > max_len:
            indices = torch.randperm(len(full_data))[:max_len]
            full_data = torch.utils.data.Subset(full_data, indices)
            print(f"Using subset {len(full_data)} images")
            

        train_size = int(0.8 * len(full_data))
        val_size = len(full_data) - train_size
        train_data, val_data = random_split(full_data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

        best_loss = float("inf")
        for epoch in range(5):
            train_stage(model, train_loader, criterion, optimizer, device, epoch)
            v_loss = val_stage(model, val_loader, criterion, device)
            scheduler.step(v_loss)

            if v_loss < best_loss:
                best_loss = v_loss
                torch.save(model.state_dict(), f"../project_numbers/checkpoints/detector_{stage}_best.pth")