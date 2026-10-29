import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """One ConvLSTM layer, Eqs. (3)-(4): one conv produces [i, f, o, g]; no peepholes."""
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, stride=1,
                              padding=k // 2, bias=True)

    def forward(self, u, state):
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([u, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ModelA(nn.Module):
    """PixelUnshuffle(4) -> 2x ConvLSTM(64, 3x3) -> 1x1 conv -> PixelShuffle(4) -> sigmoid.
    9-step rollout, no teacher forcing: steps 1-5 read the input frames, steps 6-9 read the
    model's own previous prediction. Output = predictions of steps 5..9 (5 frames)."""
    def __init__(self, patch=4, hid=64):
        super().__init__()
        self.patch, self.hid = patch, hid
        self.cell1 = ConvLSTMCell(patch * patch, hid)
        self.cell2 = ConvLSTMCell(hid, hid)
        self.out = nn.Conv2d(hid, patch * patch, kernel_size=1, bias=True)

    def forward(self, X):
        # X: (B, 5, 1, 64, 64) in [0, 1]  ->  (B, 5, 1, 64, 64)
        B = X.shape[0]
        Hh, Ww = X.shape[-2] // self.patch, X.shape[-1] // self.patch
        z = lambda: torch.zeros(B, self.hid, Hh, Ww, device=X.device, dtype=X.dtype)
        s1, s2 = (z(), z()), (z(), z())               # zero states for every video
        preds = []                                     # preds[j] = output of step j+1
        for tau in range(9):                           # tau = 0..8  <->  step 1..9
            frame = X[:, tau] if tau < 5 else preds[-1]    # own prediction fed back
            u = F.pixel_unshuffle(frame, self.patch)       # (B, 16, 16, 16)
            s1 = self.cell1(u, s1)
            s2 = self.cell2(s1[0], s2)
            xhat = torch.sigmoid(F.pixel_shuffle(self.out(s2[0]), self.patch))
            preds.append(xhat)                         # no detach: gradients flow through feedback
        return torch.stack(preds[4:], dim=1)           # outputs of steps 5-9


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
