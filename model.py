import torch
import torch.nn as nn

class LipReadingModel(nn.Module):
    def __init__(self, num_classes=1184):
        super(LipReadingModel, self).__init__()
        self.frontend = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            )
        })
        self.tcn = nn.Sequential(
            self._make_tcn(64, 64), self._make_tcn(64, 64),
            self._make_tcn(64, 64), self._make_tcn(64, 64)
        )
        self.fc = nn.Linear(64, num_classes)

    def _make_tcn(self, i, o):
        return nn.Sequential(nn.Conv1d(i, o, 3, 1, 1), nn.BatchNorm1d(o), nn.ReLU(True))

    def forward(self, x):
        # 强制换维锁
        if x.dim() == 5 and x.shape[2] == 1 and x.shape[1] != 1:
            x = x.permute(0, 2, 1, 3, 4)
        x = self.frontend['conv'](x)
        x = torch.mean(x, dim=[3, 4])
        x = self.tcn(x)
        return self.fc(torch.mean(x, dim=2))
