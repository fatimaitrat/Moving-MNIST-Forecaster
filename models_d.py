import torch, torch.nn as nn
from models import ModelA, count_params

class DilConvLSTMCell(nn.Module):
    """ConvLSTM layer with a dilated 3x3 conv; same equations (3)-(4) as model A."""
    def __init__(self, in_ch, hid_ch, k=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, stride=1,
                              padding=dilation * (k // 2), dilation=dilation, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class ModelD(ModelA):
    """Model A with layer 2 dilated (d2=3): reach 16 px. Same rollout, shapes, parameters."""
    def __init__(self, patch=4, hid=64, d1=1, d2=3):
        super().__init__(patch, hid)
        self.cell1 = DilConvLSTMCell(patch * patch, hid, 3, d1)
        self.cell2 = DilConvLSTMCell(hid, hid, 3, d2)
