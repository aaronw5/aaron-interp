"""copied from distill-part/jld-features research/discovery (same author).
Primary Lund declustering (C/A, follow the harder branch), vectorised clustering as jetdistill.observables.compute._softdrop."""
import numpy as np
from jetdistill.observables.compute import _p4, _dR2, _yphi


def lund(x, chunk=500):
    """x (J, n, 3) = (pT, Δη, Δφ). Returns per jet a list of primary emissions, hardest-angle first:
    dict(lx = ln 1/Δ, lkt = ln kt, z, soft = leaf mask of the softer branch, hard_yphi = (y, φ) of the harder branch)"""
    em = []
    for i0 in range(0, len(x), chunk):
        pt, eta, phi = (x[i0:i0 + chunk, :, c] for c in range(3)); J, H = pt.shape; rows = np.arange(J); real = pt > 0
        V = np.zeros((J, 2 * H, 4)); V[:, :H] = _p4(pt, eta, phi); leaf = np.zeros((J, 2 * H, H), bool); leaf[:, np.arange(H), np.arange(H)] = True
        kids = -np.ones((J, 2 * H, 2), int); slot = np.tile(np.arange(H), (J, 1)); act = real.copy(); iu = np.triu(np.ones((H, H), bool), 1)
        for s in range(H - 1):
            go = act.sum(1) >= 2
            if not go.any(): break
            Vs = V[rows[:, None], slot]
            Dm = np.where(act[:, :, None] & act[:, None] & iu, _dR2(Vs[:, :, None], Vs[:, None]), np.inf).reshape(J, -1)
            f = Dm.argmin(1); a, b = f // H, f % H; new = H + s; na, nb = slot[rows, a], slot[rows, b]
            V[go, new] = V[rows, na][go] + V[rows, nb][go]; kids[go, new, 0] = na[go]; kids[go, new, 1] = nb[go]
            leaf[go, new] = leaf[rows, na][go] | leaf[rows, nb][go]; slot[go, a[go]] = new; act[go, b[go]] = False
        cur = slot[rows, act.argmax(1)]; E = [[] for _ in range(J)]
        for _ in range(H):
            c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; live = c1 >= 0
            if not live.any(): break
            v1, v2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]
            p1, p2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1]); d = np.sqrt(_dR2(v1, v2))
            soft = np.where(p1 >= p2, c2, c1); hard = np.where(p1 >= p2, c1, c2); ps = np.minimum(p1, p2)
            yh, ph = _yphi(V[rows, np.maximum(hard, 0)])
            for r in np.where(live)[0]:
                E[r].append(dict(lx=float(np.log(1 / max(d[r], 1e-6))), lkt=float(np.log(max(ps[r] * d[r], 1e-6))), z=float(ps[r] / (p1[r] + p2[r])),
                                 soft=leaf[r, soft[r]].copy(), hard_yphi=(float(yh[r]), float(ph[r]))))
            cur = np.where(live, hard, cur)
        em += E
    return em
