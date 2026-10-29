import torch
from models import ModelA, count_params

torch.manual_seed(0)
m = ModelA()
print("params:", count_params(m), "(expected 480784)")
for n, mod in [("cell1", m.cell1), ("cell2", m.cell2), ("out 1x1", m.out)]:
    print(f"  {n}: {count_params(mod)}")

X = torch.rand(4, 5, 1, 64, 64)
Y = m(X)
print("output shape:", tuple(Y.shape), "| range:", Y.min().item(), Y.max().item())

# gradient flows through the fed-back predictions to every parameter
Y[:, -1].sum().backward()
print("all params got grad:", all(p.grad is not None and p.grad.abs().sum() > 0
                                  for p in m.parameters()))

# videos are independent (states reset per video)
m.eval()
with torch.no_grad():
    print("batch vs single max diff:", (m(X)[1:2] - m(X[1:2])).abs().max().item())
