import numpy as np, cv2, torch
import matplotlib.pyplot as plt
from common import *

vids = load_moving_mnist()
train, val, test = split_videos(vids)

thetas, speeds = [], []
for v in range(50):
    fr = train[v, :, 0]                                      # (20,64,64) float in [0,1]
    u8 = (fr * 255).round().to(torch.uint8).numpy()
    for s in range(19):                                      # pairs (s, s+1)
        prev, nxt = np.ascontiguousarray(u8[s]), np.ascontiguousarray(u8[s + 1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = flow[..., 0], flow[..., 1]
        mag = np.sqrt(dx**2 + dy**2)
        mask = (fr[s].numpy() > 0.2) & (fr[s + 1].numpy() > 0.2) & (mag > 0.5)
        thetas.append(np.degrees(np.arctan2(-dy[mask], dx[mask])) % 360)
        speeds.append(mag[mask])

theta = np.concatenate(thetas); speed = np.concatenate(speeds)
n_kept, med = len(theta), np.median(speed)
print("pixels kept:", n_kept, "| median speed:", med)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
w = np.ones_like(theta) / n_kept
ax[0].hist(theta, bins=np.arange(0, 361, 20), weights=w, edgecolor="k")
ax[0].axhline(1/18, color="r", ls="--", label="uniform (1/18)")
ax[0].set(xlabel="flow direction θ (deg; 0°=right, 90°=up)", ylabel="fraction of pixels",
          title="(a) direction"); ax[0].legend()
ax[1].hist(speed, bins=40, weights=np.ones_like(speed)/n_kept, edgecolor="k")
ax[1].axvline(med, color="r", label=f"median = {med:.3g}")
ax[1].set(xlabel="speed (px/frame)", ylabel="fraction of pixels", title="(b) speed"); ax[1].legend()
plt.tight_layout(); plt.savefig("figure1.png", dpi=200)
