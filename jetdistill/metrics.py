"""Accuracy, agreement with the network, and per class (one class against the other four): AUC, background rejection
1/FPR at 30 / 50 / 80 % signal efficiency (scanning the model's probability for the class), and signal efficiency and
background rejection at the model's own decision. On the whole test file."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from .config import CLASSES


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def metrics(L, y, ref=None):
    """L: class scores (J, 5); y: true class; ref: the network's class"""
    P = softmax(L); pred = L.argmax(1); out = dict(accuracy=float((pred == y).mean()), per_class={})
    if ref is not None: out['same_as_network'] = float((pred == ref).mean())
    for c in range(5):
        yy = y == c; f, tp, _ = roc_curve(yy, P[:, c]); keep = np.unique(np.linspace(0, len(f) - 1, min(len(f), 150)).astype(int))
        rej = lambda e: float(1 / max(f[min(np.searchsorted(tp, e), len(f) - 1)], 1e-7))
        out['per_class'][CLASSES[c]] = dict(op_eff=float((pred[yy] == c).mean()), op_rej=float(1 / max((pred[~yy] == c).mean(), 1e-7)),
                                            auc=float(roc_auc_score(yy, P[:, c])), rej30=rej(.3), rej50=rej(.5), rej80=rej(.8),
                                            tpr=[round(float(x), 4) for x in tp[keep]], fpr=[round(float(x), 6) for x in f[keep]])
    return out
