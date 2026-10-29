import os, numpy as np, torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from common import *
from evaluation import *          # dev, windows, per_video_mse, blank, copy_last, load_model

def fmt(x): return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:.3g}"
def ms(vals):                     # mean (seed0, seed1)
    return f"{fmt(sum(vals)/len(vals))} ({fmt(vals[0])}, {fmt(vals[1])})"

# ---------------- consistency gap and transformed error ----------------
@torch.no_grad()
def consistency(f, X, Y, T, bs=32):
    """Per-video C_T = MSE(f(T(X)), T(f(X))) and E_T = MSE(f(T(X)), T(Y)),
    each averaged over pixels and the 5 horizons of every video."""
    if hasattr(f, "eval"): f.eval()
    C, E = [], []
    for b in range(0, len(X), bs):
        xb, yb = X[b:b+bs].to(dev), Y[b:b+bs].to(dev)
        fT, Tf = f(T(xb)), T(f(xb))
        C.append(((fT - Tf) ** 2).mean(dim=(1, 2, 3, 4)).cpu())
        E.append(((fT - T(yb)) ** 2).mean(dim=(1, 2, 3, 4)).cpu())
    return torch.cat(C), torch.cat(E)

def all_metrics(f, test):
    X, Y = windows(test)
    out = {"E5": per_video_mse(f, test).mean().item()}
    for name, T in [("M", M), ("R", R)]:
        C, E = consistency(f, X, Y, T)
        out["C" + name], out["E" + name] = C.mean().item(), E.mean().item()
    return out

def table3(test):
    keys = ["E5", "CM", "EM", "CR", "ER"]
    print(" & ".join(["Model"] + keys) + r" \\")
    for name, f in [("Blank", blank), ("Copy-last", copy_last)]:
        r = all_metrics(f, test)
        print(" & ".join([name] + [fmt(r[k]) for k in keys]) + r" \\")
    A = [all_metrics(load_model("A", s), test) for s in (0, 1)]
    print(" & ".join(["A"] + [ms([A[0][k], A[1][k]]) for k in keys]) + r" \\")

# ---------------- Figure 3 panels (+ GIFs) ----------------
def make_rows(f, v, T, test):
    X, Y = windows(test)
    x, y = X[v:v+1].to(dev), Y[v:v+1].to(dev)
    with torch.no_grad():
        fx = f(x); Tx, Ty = T(x), T(y); fTx = f(Tx); Tfx = T(fx)
    c = lambda a: a[0, :, 0].cpu().numpy()                  # (5, 64, 64)
    return [(c(x), c(fx)), (c(Tx), c(Ty)), (c(Tx), c(fTx)), (c(Tx), c(Tfx))]

def draw_panel(rows, labels, title, n_vis=5):
    fig, ax = plt.subplots(4, 10, figsize=(10, 4.6))
    for r, (inp, fut) in enumerate(rows):
        for j in range(10):
            a = ax[r, j]; a.set_xticks([]); a.set_yticks([])
            img = inp[j] if j < 5 else (fut[j-5] if j-5 < n_vis else None)
            if img is None: a.axis("off")
            else: a.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[r, 0].set_ylabel(labels[r], fontsize=7, rotation=0, ha="right", va="center")
    for j in range(10):
        ax[0, j].set_title(f"in {j+1}" if j < 5 else f"h={j-4}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

def save_panel(f, v, T, tn, prefix, test, cval, who="A"):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    rows = make_rows(f, v, T, test)
    labels = ["X | f(X)", f"{tn}(X) | {tn}(Y)", f"{tn}(X) | f({tn}(X))", f"{tn}(X) | {tn}(f(X))"]
    title = f"model {who}, T={tn}: test video {v}, C_{tn}={cval:.3g}"
    fig = draw_panel(rows, labels, title, 5)
    fig.savefig(prefix + ".png", dpi=200); plt.close(fig)
    frames = []
    for h in range(1, 6):                                   # reveal horizons one by one
        fig = draw_panel(rows, labels, f"{title}  (horizon {h})", h)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames[0].save(prefix + ".gif", save_all=True, append_images=frames[1:], duration=900, loop=0)
    print("saved", prefix + ".png", "and", prefix + ".gif")

def figure3_and_4(test, f=None, who="A", out="figures"):
    f = f or load_model("A", 0)
    X, Y = windows(test)
    CM, _ = consistency(f, X, Y, M); CR, _ = consistency(f, X, Y, R)
    top = {"M": CM.topk(2).indices.tolist(), "R": CR.topk(2).indices.tolist()}
    for tn, T, C in [("M", M, CM), ("R", R, CR)]:
        for rank, v in enumerate(top[tn], 1):               # same video twice if in both lists
            save_panel(f, v, T, tn, f"{out}/fig3_{tn}{rank}", test, C[v].item(), who)
    print("top-2 C_M:", [(v, round(CM[v].item(), 4)) for v in top["M"]])
    print("top-2 C_R:", [(v, round(CR[v].item(), 4)) for v in top["R"]])
    return CM, CR, top

def figure4(CM, CR, top, fname="figure4.png"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, C, n in zip(ax, (CM, CR), ("M", "R")):
        a.hist(C.numpy(), bins=50, edgecolor="k")
        for rank, v in enumerate(top[n], 1):
            a.axvline(C[v].item(), color="r", ls="--", label=f"video {v} (rank {rank})")
        a.set(xlabel=f"per-video C_{n}", ylabel="number of test videos",
              title=f"C_{n}, seed-0 model A"); a.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(fname, dpi=200); plt.show()

# ---------------- Table 4 ----------------
def table4(test):
    cols = [("E2@1x", 1, 2), ("E2@2x", 2, 2), ("E2@3x", 3, 2), ("E5@1x", 1, 5), ("E5@2x", 2, 5)]
    e = lambda f, k, n: per_video_mse(f, test, 1, k)[:, :n].mean().item()
    print(" & ".join(["Model"] + [c[0] for c in cols]) + r" \\")
    print(" & ".join(["Copy-last"] + [fmt(e(copy_last, k, n)) for _, k, n in cols]) + r" \\")
    A = [[e(load_model("A", s), k, n) for _, k, n in cols] for s in (0, 1)]
    print(" & ".join(["A"] + [ms([A[0][i], A[1][i]]) for i in range(len(cols))]) + r" \\")

# ---------------- Figure 5 ----------------
def no_overlap_candidates(test, max_n=15):
    """Heuristic: exactly 2 connected blobs in every displayed true frame. CONFIRM VISUALLY."""
    shown = sorted({1,2,3,4,5,6,7, 1,3,5,7,9,11,13, 1,4,7,10,13,16,19})
    out = []
    for v in range(len(test)):
        ok = True
        for fr in shown:
            m = ndimage.binary_dilation(test[v, fr-1, 0].numpy() > 0.2, iterations=2)
            if ndimage.label(m)[1] != 2: ok = False; break
        if ok: out.append(v)
        if len(out) >= max_n: break
    return out

def figure5(f, v, test, fname="figure5.png"):
    if hasattr(f, "eval"): f.eval()
    fig, ax = plt.subplots(6, 7, figsize=(8.5, 7.5))
    for s, k in enumerate((1, 2, 3)):
        X, Y = windows(test[v:v+1], 1, k)
        with torch.no_grad(): P = f(X.to(dev)).cpu()
        for j in range(7):
            top_a, bot_a = ax[2*s, j], ax[2*s+1, j]
            top_a.imshow((X[0, j] if j < 5 else Y[0, j-5])[0], cmap="gray", vmin=0, vmax=1)
            fr = 1 + k*j if j < 5 else 1 + 5*k + k*(j-5)
            top_a.set_title(f"frame {fr}", fontsize=7)
            if j >= 5: bot_a.imshow(P[0, j-5, 0], cmap="gray", vmin=0, vmax=1)
            for a in (top_a, bot_a): a.set_xticks([]); a.set_yticks([])
            if j < 5: bot_a.axis("off")
        ax[2*s, 0].set_ylabel(f"{k}x\ntruth", fontsize=8)
        ax[2*s+1, 0].text(0.5, 0.5, f"{k}x\nprediction (h=1,2 at right)", ha="center", va="center", fontsize=7)
    fig.suptitle(f"test video {v}: inputs (cols 1-5), then true / predicted h=1,2", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(fname, dpi=200); plt.show()
