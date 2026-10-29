from section3 import *
from models_d import ModelD
from sym import Sym

_, _, test = split_videos(load_moving_mnist())
f = Sym(load_model("D", 0, ModelD))            # final model, seed 0

# (a) Figure 3 videos, exactly as before: M -> 292, 995; R -> 168, 36
X, Y = windows(test)
for tn, T, vids in [("M", M, [292, 995]), ("R", R, [168, 36])]:
    C, _ = consistency(f, X, Y, T)
    for rank, v in enumerate(vids, 1):
        save_panel(f, v, T, tn, f"figures6/fig6a_{tn}{rank}", test, C[v].item(), "final (sym D)")

# (b) Figure 5 video
figure5(f, 29, test, "figure6b.png")
