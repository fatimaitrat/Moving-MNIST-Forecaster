# Q2.5 prediction (written before any Section 3 experiment)
Mirror test: I expect model A to partly pass. Mirrored trajectories are largely familiar (Fig. 1a is roughly balanced across directions), so the digits should move in roughly the right way. The digit shapes are mirrored and unfamiliar, so I expect some blurring or distortion, and a noticeably non-zero $C_M$ (consistency gap).

Rotation test: I expect it to do worse than mirror. The motion statistics are again mostly familiar, but rotated digits look even less like the training digits, so I expect bigger errors and a larger $C_R$ than $C_M$.

Speed: the reach estimate is 8 px/frame (Q2.3), and the typical training speed is 3.3 px/frame with nothing above about 5 (Fig. 1b). At $2\times$ ($\approx6.6$ px/frame) the motion is inside the reach but beyond all training speeds, so I expect degraded but partly working predictions, such as blur or duplicated digits. At $3\times$ ($\approx10$ px/frame) the motion is beyond both the reach and the training speeds, so I expect a clear breakdown. My prediction is that model A breaks down between $2\times$ and $3\times$, and fails clearly at $3\times$.
