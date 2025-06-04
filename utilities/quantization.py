import numpy as np

def quantize(x, Delta):
    if Delta is None:
        return x
    return Delta * np.round(x / Delta)