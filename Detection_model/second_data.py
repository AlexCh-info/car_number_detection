import os
import cv2
import torch
from numba.cpython.unicode import gen_unicode_Xjust
from torch.utils.data import Dataset
from pathlib import Path

class DataNumber(Dataset):
    def __init__(self, root_dir, grid_size=10, img_size=320):
        self.root_dir = Path(root_dir)
        self.grid_size = grid_size
        self.img_size = img_size

        self.img_paths = list(self.root_dir.rglob('*.jpg'))
        print(f"Найдено изображений {len(self.img_paths)}")

    def __len__(self):
        return len(self.img_paths)

    def parse_name(self, name):
        parts = name.split('-')
        bounding_box = parts[2]
        p1, p2 = bounding_box.split("_")
        x1, y1 = map(int, p1.split("&"))
        x2, y2 = map(int, p2.split("&"))
        return x1, y1, x2, y2


    def __getitem__(self, item):
        img_path = self.img_paths[item]

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self.__getitem__((item + 1) % len(self))

        h_orig, w_orig = img.shape
        img = cv2.resize(img, (self.img_size, self.img_size))

        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        try:
            x1, y1, x2, y2 = self.parse_name(img_path.stem)
        except:
            return self.__getitem__((item + 1) % len(self))

        target = torch.zeros((self.grid_size, self.grid_size, 5))

        xc = (x1 + x2) / 2 / w_orig
        yc = (y1 + y2) / 2 / h_orig
        nw = (x2 - x1) / w_orig
        nh = (y2 - y1) / h_orig

        gx = int(xc * self.grid_size)
        gy = int(yc * self.grid_size)

        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            target[gy, gx, 0] = 1.0 # Confidence
            target[gy, gx, 1] = xc * self.grid_size - gx # X
            target[gy, gx, 2] = yc * self.grid_size - gy # Y
            target[gy, gx, 3] = nw # W
            target[gy, gx, 4] = nh # H
        return img_tensor, target


