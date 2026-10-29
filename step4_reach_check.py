import torch, torch.nn.functional as F
from models import ModelA, count_params
from models_d import ModelD

def measured_reach_px(model, patch=4):
    """Impulse in input frame 5 (step 5), pixel (30,30) = cell (7,7). Look at how far the
    change in the first forecast frame spreads (same-step paths only)."""
    model = model.double().eval()
    X = torch.zeros(1, 5, 1, 64, 64, dtype=torch.double)
    X2 = X.clone(); X2[0, 4, 0, 30, 30] = 1.0
    with torch.no_grad():
        d = model(X2)[0, 0] - model(X)[0, 0]                     # (1,64,64)
    cells = F.pixel_unshuffle(d, patch).abs().amax(dim=0)         # (16,16)
    nz = (cells > 1e-12).nonzero()
    c0 = 30 // patch
    return (nz - c0).abs().max().item() * patch

torch.manual_seed(0)
for name, m in [("A", ModelA()), ("D", ModelD())]:
    print(name, "params:", count_params(m), "| measured reach (px):", measured_reach_px(m))
