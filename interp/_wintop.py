"""copied from distill-part/jld-features research/discovery (same author).
Shared edit for G199 / J30: move the W-candidate mass inside a top jet at fixed top mass (pT-weighted k-means, 3 subjets).
Works on (pt, deta, dphi) arrays (J, n): returns new deta, dphi, ok mask, and the W-pair / jet masses after."""
import numpy as np
from jetdistill.observables.compute import _kmeans, _p4, _m4


def masses(pt, de, dp, sel=None):
    w = pt if sel is None else pt * sel; return _m4(*_p4(w, de, dp).sum(1).T)


def move_w(pt, de, dp, target):
    ax, d = _kmeans(pt, de, dp, 3); a = d.argmin(-1); a[pt <= 0] = -1
    v = np.stack([_p4(pt * (a == j), de, dp).sum(1) for j in range(3)], 1); P = [(0, 1), (0, 2), (1, 2)]
    pm = np.stack([_m4(*(v[:, i] + v[:, j]).T) for i, j in P], 1); w = np.abs(pm - 80.4).argmin(1)
    ok = np.abs(pm[np.arange(len(pt)), w] - 80.4) < 15
    inW = np.zeros(pt.shape, bool)
    for r in range(len(pt)): inW[r] = np.isin(a[r], P[w[r]])
    m_top = masses(pt, de, dp); de, dp = de.copy(), dp.copy()
    for _ in range(5):
        wp = pt * inW; c = (wp[..., None] * np.stack([de, dp], -1)).sum(1) / np.maximum(wp.sum(1), 1e-9)[:, None]
        f = (target / np.maximum(masses(pt, de, dp, inW), 1e-3))[:, None]
        de = np.where(inW, c[:, None, 0] + f * (de - c[:, None, 0]), de); dp = np.where(inW, c[:, None, 1] + f * (dp - c[:, None, 1]), dp)
        s = (m_top / np.maximum(masses(pt, de, dp), 1e-3))[:, None]; de, dp = de * s, dp * s
    return de, dp, ok, masses(pt, de, dp, inW), masses(pt, de, dp)
