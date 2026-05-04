import json
import cv2
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CHARS = "0123456789АВЕКМНОРСТУХ"
LATIN_TO_CYR = {
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е',
    'H': 'Н', 'K': 'К', 'M': 'М',
    'O': 'О', 'P': 'Р', 'T': 'Т', 'X': 'Х', 'Y': 'У'
} # v2 - Если в тексте есть латиница, то меняем на кириллицу.
char2idx = {c: i + 1 for i, c in enumerate(CHARS)}


class NumeroffData(Dataset):
    def __init__(self, img_dir: str | Path, json_path: str | Path):
        self.img_dir = Path(img_dir)
        self.json_path = Path(json_path)
        self.samples = []

        self.augment = A.Compose([
            A.Resize(32, 128),
            A.OneOf([
                A.MotionBlur(5),
                A.GaussianBlur(5),
            ], p=0.3), # v2 - чуть-чуть меняем вид аугментации.
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.CLAHE(),
            ], p=0.4),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.Affine(
                rotate=(-5, 5),
                shear=(-5, 5),
                p=0.3
            ),
            A.Downscale(p=0.3), # v3 - снижаем качество изображения.
            A.CoarseDropout(num_holes_range=(1,4), p=0.2),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

        # Собираем все аннотации из папки
        for ann_file in self.json_path.iterdir():
            if ann_file.suffix.lower() not in ['.json']:
                continue
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                img_file = self.img_dir / f"{data['name']}.png"
                text = self.normalize_text(data["description"]) # v2 - делаем замену.
                encoded = self.encode(text)
                if len(encoded) ==0:
                    continue
                self.samples.append((img_file, torch.tensor(encoded, dtype=torch.long)))

    @staticmethod
    def encode(text):
        return [char2idx[c] for c in text if c in char2idx]

    @staticmethod
    def normalize_text(text): # v2 - функция замены символов.
        return ''.join(LATIN_TO_CYR.get(c, c) for c in text)

    def __len__(self):  # Исправлено имя метода
        return len(self.samples)

    def __getitem__(self, item):
        img_path, target = self.samples[item]  # берем уже готовый target.

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Broken img -> {img_path.name}")

        image_tensor = self.augment(image=img)['image']
        return image_tensor, target

if __name__ == "__main__":
    data = NumeroffData(img_dir='../data/autoriaNumberplateOcrRu-2021-09-01/train/img',
                        json_path='../data/autoriaNumberplateOcrRu-2021-09-01/train/ann')