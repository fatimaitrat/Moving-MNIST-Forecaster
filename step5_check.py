import torch
from common import *
from evaluation import dev, load_model
from section3 import consistency
from sym import Sym

f = load_model("A", 0)
X, Y = torch.rand(8, 5, 1, 64, 64), torch.rand(8, 5, 1, 64, 64)
for n, T in [("M", M), ("R", R)]:
    print(n, "| plain f:", consistency(f, X, Y, T)[0].max().item(),
             "| sym f:", consistency(Sym(f), X, Y, T)[0].max().item())
