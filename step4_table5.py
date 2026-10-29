import json, torch
from common import *
from evaluation import *
from section3 import all_metrics
from models import ModelA, count_params
from models_d import ModelD

FN = {"A": ModelA, "B": ModelA, "C": ModelA, "D": ModelD}

def fmt(x):   return f"{x:.2e}" if abs(x) < 1e-3 else f"{x:#.3g}"
def ms(v, f=fmt): return f"{f(sum(v)/len(v))} ({f(v[0])}, {f(v[1])})"
t1 = lambda x: f"{x:.1f}"
t2 = lambda x: f"{x:.2f}"

_, _, test = split_videos(load_moving_mnist())
e2 = lambda f, k: per_video_mse(f, test, 1, k)[:, :2].mean().item()

def collect(tag):
    out = []
    for s in (0, 1):
        f = load_model(tag, s, FN[tag])
        r = all_metrics(f, test)
        for k in (1, 2, 3): r[f"E2@{k}x"] = e2(f, k)
        r["params"] = count_params(f)
        r["train_s"] = json.load(open(f"ckpt/{tag}_seed{s}.json"))["mean_epoch_time"]
        inference_time(f, test[:64])                  # warm-up pass, not timed
        r["infer_s"] = inference_time(f, test)
        out.append(r)
    return out

res = {t: collect(t) for t in "ABCD"}
json.dump(res, open("table5_raw.json", "w"), indent=1)

keys = ["E5", "CM", "EM", "CR", "ER", "E2@1x", "E2@2x", "E2@3x"]
print(" & ".join(["Model"] + keys + ["Params", "Train s/ep", "Infer s"]) + r" \\")
for t, rows in res.items():
    cells = [ms([r[k] for r in rows]) for k in keys]
    cells += [str(rows[0]["params"]), ms([r["train_s"] for r in rows], t1), ms([r["infer_s"] for r in rows], t2)]
    print(" & ".join([t] + cells) + r" \\")
