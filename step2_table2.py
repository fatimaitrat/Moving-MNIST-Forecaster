import torch
from common import *
from evaluation import *

_, _, test = split_videos(load_moving_mnist())

def fmt(x): return f"{x:.2e}" if x < 1e-3 else f"{x:.3g}"
def row(mse):                          # mse: (N,5) -> MSE_1..5 and E5
    h = mse.mean(0); return [h[i].item() for i in range(5)] + [mse.mean().item()]

det = {"Blank": row(per_video_mse(blank, test)), "Copy-last": row(per_video_mse(copy_last, test))}
seeds = {s: row(per_video_mse(load_model("A", s), test)) for s in (0, 1)}
params = sum(p.numel() for p in load_model("A", 0).parameters())

cols = ["MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E5"]
print(" & ".join(["Model"] + cols + ["Params"]))
for name, r in det.items():
    print(" & ".join([name] + [fmt(v) for v in r] + ["0"]) + r" \\")
a = [f"{fmt((seeds[0][i]+seeds[1][i])/2)} ({fmt(seeds[0][i])}, {fmt(seeds[1][i])})" for i in range(6)]
print(" & ".join(["A"] + a + [str(params)]) + r" \\")
