import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from first_model import OCRModel
from data import NumeroffData, CHARS

def init_weights(m):
    """Т.к. модель обучается с нуля хорошим бустом для нее будет определение весов."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)

    # Склеиваем все таргеты в один плоский тензор (как любит CTCLoss)
    # или используйте pad_sequence, если ваша модель ждет матрицу
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets = torch.cat(targets)

    return images, targets, target_lengths

def train(epoch):
    model.train()
    tr_loss = 0
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}", leave=True)
    for batch_idx,(imgs, targets, target_length) in pbar:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        # 1. Получаем выход модели (Batch, Time, Classes)
        outputs = model(imgs)

        # 2. Переставляем оси для CTCLoss -> (Time, Batch, Classes)
        outputs = outputs.permute(1, 0, 2)

        # 3. Теперь правильно берем размерности
        time_steps = outputs.size(0)
        current_batch_size = outputs.size(1)

        input_lengths = torch.full(size=(current_batch_size,),
                                   fill_value=time_steps,
                                   dtype=torch.long).to(device)
        target_length = target_length.to(device)

        loss = criterion(outputs, targets, input_lengths, target_length)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        pbar.set_postfix({"loss": f"{tr_loss / (batch_idx + 1):.4f}"})

def val():
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (imgs, targets, target_length) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validation", leave=False):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            outputs = outputs.permute(1, 0, 2)

            time_steps = outputs.size(0)
            current_batch_size = outputs.size(1)

            input_lengths = torch.full(size=(current_batch_size,),
                                       fill_value=time_steps,
                                       dtype=torch.long).to(device)
            target_length = target_length.to(device)

            loss = criterion(outputs, targets, input_lengths, target_length)
            test_loss += loss.item()
    avg_loss = test_loss / len(val_dataloader)
    print(f"Test loss: {avg_loss:.3f}")
    return avg_loss


if __name__ == '__main__':
    logger = logging.getLogger(__name__)  # Создаем logger.
    logging.basicConfig(level=logging.INFO)  # Включаем отображение логов.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Автоматический выбор девайса.
    model = OCRModel(len(CHARS) + 1)  # Создаем модель.
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # Выбираем лосс функцию.
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # Выбираем оптимизатор обучения.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Выбираем пока один callback.

    # Создаем выборки данных.
    train_data = NumeroffData('../data/autoriaNumberplateOcrRu-2021-09-01/train/img',
                              '../data/autoriaNumberplateOcrRu-2021-09-01/train/ann')
    val_data = NumeroffData('../data/autoriaNumberplateOcrRu-2021-09-01/val/img',
                            '../data/autoriaNumberplateOcrRu-2021-09-01/val/ann')

    train_dataloader = DataLoader(train_data, 64, True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, 64, False, collate_fn=collate_fn)

    model.apply(init_weights)
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(20):
        train(epoch)
        val_loss = val()
        scheduler.step(val_loss)

        torch.save(model.state_dict(), 'checkpoints/last_model_edit_v3.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model_edit_v3.pth')
            print(f"Модель сохранена (val_loss: {best_val_loss:.4f})")