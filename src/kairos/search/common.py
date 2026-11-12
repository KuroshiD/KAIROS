"""Constantes e funcoes numericas compartilhadas pela busca."""
import time

import numpy as np

CLIP = 1000.0
INF = float("inf")


def log(msg=""):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def clip(v):
    return float(max(-CLIP, min(CLIP, v)))


def logsumexp(x):
    x = np.asarray(x, float)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def softmax(x):
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()
