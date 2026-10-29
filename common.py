import torch, torchvision

# ---------- data ----------
def load_moving_mnist(root="data"):
    ds = torchvision.datasets.MovingMNIST(root=root, split=None, download=True)
    vids = torch.stack([ds[i] for i in range(len(ds))])      # (10000, 20, 1, 64, 64)
    vids = vids.float() / 255.0
    assert vids.shape == (10000, 20, 1, 64, 64), vids.shape
    return vids

def split_videos(vids):
    return vids[:8000], vids[8000:9000], vids[9000:10000]    # train, val, test

# ---------- symmetries (work on (..., H, W) tensors, any leading dims) ----------
def M(x):   return torch.flip(x, dims=(-1,))                 # (Mx)[i,j] = x[i,63-j]
def R(x, k=1): return torch.rot90(x, k=k, dims=(-2, -1))     # 90° anticlockwise, k times
def MR(x, a): return M(R(x, a))                              # rotate first, then mirror

# G as (name, forward, inverse); R^a inverse is R^-a, M R^a is its own inverse
G = []
for a in range(4):
    G.append((f"R{a}",  (lambda x, a=a: R(x, a)),   (lambda x, a=a: R(x, -a))))
for a in range(4):
    G.append((f"MR{a}", (lambda x, a=a: MR(x, a)),  (lambda x, a=a: MR(x, a))))

# ---------- windows (frames are 1-indexed, as in the handout) ----------
def get_window(video, t, k=1):
    """video: (20,1,64,64). Returns X (5,...) and Y (n<=5,...)."""
    in_idx  = [t + k * j for j in range(5)]                  # t, t+k, ..., t+4k
    tgt_idx = [t + 5 * k + k * j for j in range(5)]          # t+5k, ..., t+9k
    tgt_idx = [f for f in tgt_idx if f <= 20]                # keep only existing frames
    assert max(in_idx) <= 20
    X = video[[f - 1 for f in in_idx]]
    Y = video[[f - 1 for f in tgt_idx]]
    return X, Y, in_idx, tgt_idx

def allowed_starts(k, min_targets=2):
    """Starts where all 5 inputs and >= min_targets targets exist (augmented training).
    Gives t in 1..14 (k=1), 1..8 (k=2), 1..2 (k=3)."""
    return [t for t in range(1, 21)
            if t + 4 * k <= 20 and t + 5 * k + (min_targets - 1) * k <= 20]

def allowed_starts_normal():
    return list(range(1, 12))                                # models A, B: t = 1..11
