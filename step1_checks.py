import torch
from common import *

torch.manual_seed(0); x = torch.rand(20, 1, 64, 64)
p = torch.zeros(20, 1, 64, 64); p[7, 0, 10, 20] = 1

def report(name, a, b):
    print(f"{name}: max|diff| = {(a-b).abs().max().item()}, equal = {torch.equal(a,b)}")

report("(a) M(M(x)) vs x",       M(M(x)), x)
report("(b) R^4(x) vs x",        R(x, 4), x)
report("(c) M R M (x) vs R^-1",  M(R(M(x))), R(x, -1))

def bright(t):
    idx = (t == 1).nonzero()[0].tolist()
    return (idx[0], idx[2], idx[3])                          # (frame, row, col)
print("(d) M(p):", bright(M(p)), "expected (7, 10, 43)")
print("(e) R(p):", bright(R(p)), "expected (7, 43, 10)")

# frame f (1-indexed) filled with value f
v = torch.arange(1, 21).float().view(20, 1, 1, 1).expand(20, 1, 64, 64).clone()
for k, name in [(1, "(f) 1x"), (2, "(g) 2x"), (3, "(h) 3x")]:
    X, Y, _, _ = get_window(v, t=1, k=k)
    print(name, "input", X[:, 0, 0, 0].int().tolist(), "target", Y[:, 0, 0, 0].int().tolist())

# extra sanity: allowed starts
print({k: allowed_starts(k) for k in (1, 2, 3)})
