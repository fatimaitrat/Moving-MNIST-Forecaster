import sys
from train import train
from models import ModelA
from models_d import ModelD

CFG = {
    "B":  dict(model_fn=ModelA, ks=(1,),    min_targets=5, sym=True),
    "C":  dict(model_fn=ModelA, ks=(1,2,3), min_targets=2, sym=False),
    "D":  dict(model_fn=ModelD, ks=(1,2,3), min_targets=2, sym=False),
    "DB": dict(model_fn=ModelD, ks=(1,2,3), min_targets=2, sym=True),   # optional
}
if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]), **CFG[sys.argv[1]])
