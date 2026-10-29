import time, torch
from common import *
from models import ModelA

dev = "cuda" if torch.cuda.is_available() else "cpu"

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def windows(vids, t=1, k=1):
    """vids: (N,20,1,64,64) -> X (N,5,...), Y (N,n<=5,...) for start frame t, speed k."""
    ar = torch.arange(5)
    in_i = (t - 1) + k * ar
    tg = (t - 1) + 5 * k + k * ar
    tg = tg[tg <= 19]
    return vids[:, in_i], vids[:, tg]

@torch.no_grad()
def per_video_mse(f, vids, t=1, k=1, bs=32):
    """Returns (N, n) tensor: MSE_h of every video (mean over the 64x64 pixels)."""
    if hasattr(f, "eval"): f.eval()
    X, Y = windows(vids, t, k)
    n = Y.shape[1]
    out = []
    for b in range(0, len(X), bs):
        xb, yb = X[b:b + bs].to(dev), Y[b:b + bs].to(dev)
        p = f(xb)[:, :n]
        out.append(((p - yb) ** 2).mean(dim=(2, 3, 4)).cpu())
    return torch.cat(out)

blank = lambda X: torch.zeros_like(X)
copy_last = lambda X: X[:, -1:].expand(-1, 5, -1, -1, -1)

@torch.no_grad()
def inference_time(f, vids, bs=32):
    """Wall-clock time to predict all test videos (1x, t=1), batch on device first."""
    if hasattr(f, "eval"): f.eval()
    X, _ = windows(vids)
    total = 0.0
    for b in range(0, len(X), bs):
        xb = X[b:b + bs].to(dev)
        sync(); t0 = time.time()
        f(xb)
        sync(); total += time.time() - t0
    return total

def load_model(tag, seed, model_fn=ModelA, ckpt_dir="ckpt"):
    m = model_fn().to(dev)
    m.load_state_dict(torch.load(f"{ckpt_dir}/{tag}_seed{seed}.pt", map_location=dev))
    return m.eval()
