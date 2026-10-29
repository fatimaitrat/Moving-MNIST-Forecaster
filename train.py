import time, json, os, sys
import torch
from common import *
from models import ModelA, count_params
from evaluation import dev, sync, per_video_mse

def masked_loss(pred, Y, mask):
    """pred, Y: (B,5,1,64,64); mask: (B,5). MSE per video over pixels and available
    target frames, then mean over the videos in the batch."""
    mse = ((pred - Y) ** 2).mean(dim=(2, 3, 4))              # (B,5)
    return ((mse * mask).sum(1) / mask.sum(1)).mean()

def make_batch(vids, idx, g, ks=(1,), min_targets=5, sym=False):
    Xs, Ys, Ms = [], [], []
    ar = torch.arange(5)
    for v in idx.tolist():
        k = ks[torch.randint(len(ks), (1,), generator=g).item()]
        starts = allowed_starts(k, min_targets)               # min_targets=5 -> t=1..11 at 1x
        t = starts[torch.randint(len(starts), (1,), generator=g).item()]
        in_i = (t - 1) + k * ar
        tg_i = (t - 1) + 5 * k + k * ar
        m = (tg_i <= 19).float()
        X = vids[v, in_i.to(vids.device)]
        Y = vids[v, tg_i.clamp(max=19).to(vids.device)]
        if sym:
            gi = torch.randint(8, (1,), generator=g).item()
            X, Y = G[gi][1](X), G[gi][1](Y)
        Xs.append(X); Ys.append(Y); Ms.append(m)
    return torch.stack(Xs), torch.stack(Ys), torch.stack(Ms).to(vids.device)

def train(tag, seed, model_fn=ModelA, ks=(1,), min_targets=5, sym=False,
          epochs=30, bs=32, ckpt_dir="ckpt"):
    os.makedirs(ckpt_dir, exist_ok=True)
    tr, va, _ = split_videos(load_moving_mnist())
    tr, va = tr.to(dev), va.to(dev)
    torch.manual_seed(seed)                                   # controls init
    model = model_fn().to(dev)
    print(f"[{tag} seed {seed}] params = {count_params(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)       # no schedule
    g = torch.Generator().manual_seed(seed)                   # shuffling, windows, augmentation
    epoch_times, log = [], []
    for ep in range(epochs):
        model.train(); sync(); t0 = time.time()
        perm = torch.randperm(len(tr), generator=g)
        run = 0.0
        for b in range(0, len(tr), bs):
            X, Y, mask = make_batch(tr, perm[b:b + bs], g, ks, min_targets, sym)
            loss = masked_loss(model(X), Y, mask)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(X)
        sync(); epoch_times.append(time.time() - t0)
        val = per_video_mse(model, va).mean().item()          # monitoring only (after the timer)
        log.append({"epoch": ep + 1, "train_loss": run / len(tr), "val_E5": val})
        print(f"ep {ep+1:2d}  train {run/len(tr):.5f}  val E5 {val:.5f}  {epoch_times[-1]:.1f}s")
    torch.save(model.state_dict(), f"{ckpt_dir}/{tag}_seed{seed}.pt")   # final-epoch weights
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    json.dump({"epoch_times": epoch_times, "mean_epoch_time": sum(epoch_times) / len(epoch_times),
               "device": gpu, "log": log}, open(f"{ckpt_dir}/{tag}_seed{seed}.json", "w"))

if __name__ == "__main__":
    tag, seed = sys.argv[1], int(sys.argv[2])
    assert tag == "A"
    train("A", seed)
