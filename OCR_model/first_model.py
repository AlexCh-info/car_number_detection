import torch
import torch.nn as nn

class DSConv(nn.Module):
    def __init__(self, in_c, out_c, stride=(1, )):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.block(x)

# v4 - Residual-block
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))



"""class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            DSConv(32, 64, 2),
            DSConv(64, 128, 2),
            DSConv(128, 256, 2),
            DSConv(256, 256, stride=(2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.rnn = nn.LSTM(
            input_size=1024, # v1 - 512, v2 - 512, v3 - 1024 (т.к. используем view)
            hidden_size=384, # v1 - 256, v2 - 384, v3 - 384 (увеличили для повышения качетсва)
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3) # v2 - Добавили для устойчивости модели

        self.head = nn.Linear(768, num_classes) # v1 - 512, v2 - 768 (т.к. в lstm используем bidirectional)

    def forward(self, x):
        x = self.backbone(x) # B, C, H, W
        b, c, h, w = x.size()
        # collapse height (B, C, H, W) -> (B, C * H, W)
        x = x.view(b, c * h, w) # B, C, W v2 - используем view чтобы сохранить важные признаки
        x = x.permute(0, 2, 1) # B, W, C все версии меняем местами

        weights = self.attention(x) # v3 - применяем слой внимания
        x = x * weights # v3 - выделяем признаки

        x, _ = self.rnn(x)
        x = self.dropout(x) # v2 - прменяем dropout

        x = self.head(x)
        x = x.log_softmax(2)

        return x"""

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, stride=(2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True), # v4 - Меняем тангенс на ReLU
            nn.Linear(512, 1024), # v4 - Возвращаем размерность, чтобы сделать более точный attention/
            nn.Sigmoid() # v4 - Меняем Softmax на Sigmoid
        )

        self.rnn = nn.LSTM(
            input_size=1024, # v1 - 512, v2 - 512, v3 - 1024 (т.к. используем view)
            hidden_size=384, # v1 - 256, v2 - 384, v3 - 384 (увеличили для повышения качетсва)
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3) # v2 - Добавили для устойчивости модели

        self.head = nn.Linear(768, num_classes) # v1 - 512, v2 - 768 (т.к. в lstm используем bidirectional)

    def forward(self, x):
        x = self.backbone(x) # B, C, H, W
        b, c, h, w = x.size()
        # collapse height (B, C, H, W) -> (B, C * H, W)
        x = x.view(b, c * h, w) # B, C, W v2 - используем view чтобы сохранить важные признаки
        x = x.permute(0, 2, 1) # B, W, C все версии меняем местами

        weights = self.attention(x) # v3 - применяем слой внимания
        x = x * weights # v3 - выделяем признаки

        x, _ = self.rnn(x)
        x = self.dropout(x) # v2 - прменяем dropout

        x = self.head(x)
        x = x.log_softmax(2)

        return x
