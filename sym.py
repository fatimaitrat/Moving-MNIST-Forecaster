from common import G

class Sym:
    """Symmetrised forecaster, Eq. (7): (1/8) * sum_g g^-1( f( g(X) ) ). No retraining."""
    def __init__(self, f): self.f = f
    def eval(self): self.f.eval(); return self
    def __call__(self, X):
        out = 0
        for _, g, ginv in G:
            out = out + ginv(self.f(g(X)))
        return out / 8
