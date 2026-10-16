"""Vectorised numpy implementation of every observable of library.py (same algorithms and tie-breaking as the
plain-Python code of the exported files; tests/test_observables.py compares them).

compute(X, n) -> {id: array (J,)} for particles X (J, n, 3) = (pT [GeV], Δη, Δφ), hardest first, padding pT = 0 at the end."""
import itertools
import numpy as np
from .library import KH, KS, TOPK, RINGS, ring_tag, library

H3, H4, HSD = 24, 12, 20   # ECF e2/e3 on the first 24 slots, e4 on the first 12, soft drop on the first 20
ZCUT, R0, EPS = 0.1, 0.8, 1e-30


def compute(X, n, ids=None, chunk=1000):
    """all observables (or only `ids`) of the jets X, in chunks of `chunk` jets"""
    X = np.asarray(X, np.float64).reshape(len(X), n, 3); out = {}
    for i0 in range(0, len(X), chunk):
        for k, v in _chunk(X[i0:i0 + chunk], n).items():
            if ids is None or k in ids: out.setdefault(k, []).append(v)
    O = {k: np.nan_to_num(np.concatenate(v)) for k, v in out.items()}
    lib = library(n); missing = set(ids or lib) - set(O)
    assert not missing, f'no numpy implementation for {sorted(missing)}'
    return {k: O[k] for k in (ids or lib)}


def _m4(px, py, pz, E):
    return np.sqrt(np.maximum(E * E - px * px - py * py - pz * pz, 0.0))


def _p4(pt, eta, phi):
    return np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1)


def _kmeans(pt, eta, phi, k, iters=6):
    """farthest-point initialisation (pT-weighted) then 6 pT-weighted k-means updates in (Δη, Δφ).
    Returns the axes (J, k, 2) and the distance of every particle to every axis (J, n, k)."""
    J = len(pt); ax = np.zeros((J, k, 2)); pts = np.stack([eta, phi], -1)
    ax[:, 0] = pts[np.arange(J), pt.argmax(1)]
    for j in range(1, k):
        d = np.min(np.linalg.norm(pts[:, :, None] - ax[:, None, :j], axis=-1), -1) * pt
        ax[:, j] = pts[np.arange(J), d.argmax(1)]
    for _ in range(iters):
        a = np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1).argmin(-1)
        for j in range(k):
            w = pt * (a == j); s = w.sum(1, keepdims=True)
            ax[:, j] = np.where(s > 0, (w[..., None] * pts).sum(1) / np.maximum(s, 1e-12), ax[:, j])
    return ax, np.linalg.norm(pts[:, :, None] - ax[:, None], axis=-1)


def _yphi(v):
    return 0.5 * np.log(np.maximum(v[..., 3] + v[..., 2], 1e-300) / np.maximum(v[..., 3] - v[..., 2], 1e-300)), np.arctan2(v[..., 1], v[..., 0])


def _dR2(u, v):
    (ya, pa), (yb, pb) = _yphi(u), _yphi(v); dp = np.mod(pa - pb + np.pi, 2 * np.pi) - np.pi
    return (ya - yb) ** 2 + dp ** 2


def _softdrop(pt, eta, phi):
    """Cambridge/Aachen clustering (E-scheme, rapidity-azimuth distance) of the real particles among the first HSD slots,
    then soft drop (β = 0, z_cut = 0.1): follow the harder branch until min(pT1, pT2)/(pT1 + pT2) > z_cut.
    Returns the groomed mass, z_g, R_g and the number of removed branches."""
    J, H = pt.shape; real = pt > 0; rows = np.arange(J)
    V = np.zeros((J, 2 * H, 4)); V[:, :H] = _p4(pt, eta, phi)
    kids = -np.ones((J, 2 * H, 2), int); slot_node = np.tile(np.arange(H), (J, 1)); active = real.copy()
    iu = np.triu(np.ones((H, H), bool), 1)
    for s in range(H - 1):
        go = active.sum(1) >= 2
        if not go.any(): break
        Vs = V[rows[:, None], slot_node]
        D = np.where(active[:, :, None] & active[:, None] & iu, _dR2(Vs[:, :, None], Vs[:, None]), np.inf).reshape(J, -1)
        f = D.argmin(1); a, b = f // H, f % H
        new = H + s; na, nb = slot_node[rows, a], slot_node[rows, b]
        V[go, new] = V[rows, na][go] + V[rows, nb][go]; kids[go, new, 0] = na[go]; kids[go, new, 1] = nb[go]
        slot_node[go, a[go]] = new; active[go, b[go]] = False
    cur = slot_node[rows, active.argmax(1)]; done = np.zeros(J, bool)
    mass, zg, rg, nrm = np.zeros(J), np.zeros(J), np.zeros(J), np.zeros(J)
    for _ in range(H):
        c1, c2 = kids[rows, cur, 0], kids[rows, cur, 1]; done |= c1 < 0
        if done.all(): break
        v1, v2 = V[rows, np.maximum(c1, 0)], V[rows, np.maximum(c2, 0)]
        p1, p2 = np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1]); zz = np.minimum(p1, p2) / np.maximum(p1 + p2, 1e-300)
        stop = ~done & (zz > ZCUT); vc = V[rows, cur]
        mass = np.where(stop, _m4(*vc.T), mass); zg = np.where(stop, zz, zg); rg = np.where(stop, np.sqrt(_dR2(v1, v2)), rg)
        done |= stop; step = ~done
        nrm += step; cur = np.where(step, np.where(p1 >= p2, c1, c2), cur)
    return mass, zg, rg, nrm


def _ecf(zt, et, pht):
    """e2, e2(β=2), e3, e3(β=2), ₁e₃, ₂e₃ of the given particles, their distance matrix and pT shares"""
    h = zt.shape[1]; R = np.sqrt((et[:, :, None] - et[:, None]) ** 2 + (pht[:, :, None] - pht[:, None]) ** 2)
    iu = np.triu_indices(h, 1); w2 = zt[:, iu[0]] * zt[:, iu[1]]; rr = R[:, iu[0], iu[1]]
    I, J, K = np.array(list(itertools.combinations(range(h), 3))).T
    a, b, c = R[:, I, J], R[:, I, K], R[:, J, K]; w3 = zt[:, I] * zt[:, J] * zt[:, K]; s3 = np.sort(np.stack([a, b, c], -1), -1)
    return ((w2 * rr).sum(1), (w2 * rr ** 2).sum(1), (w3 * a * b * c).sum(1), (w3 * (a * b * c) ** 2).sum(1),
            (w3 * s3[..., 0]).sum(1), (w3 * s3[..., 0] * s3[..., 1]).sum(1), R, zt)


def _chunk(X, n):
    pt, eta, phi = X[..., 0], X[..., 1], X[..., 2]; J = len(X); rows = np.arange(J)
    real = pt > 0; nreal = real.sum(1); tot = pt.sum(1); z = pt / tot[:, None]; dr = np.hypot(eta, phi)
    P4 = _p4(pt, eta, phi)
    mass_of = lambda k: _m4(*P4[:, :k].sum(1).T)
    dist = lambda i, j: np.hypot(eta[:, i] - eta[:, j], phi[:, i] - phi[:, j])
    pm = lambda i, j: np.sqrt(np.maximum(2 * pt[:, i] * pt[:, j] * (np.cosh(eta[:, i] - eta[:, j]) - np.cos(phi[:, i] - phi[:, j])), 0))
    O = {}
    # ---- whole jet ----
    m = mass_of(n); zs = -np.sort(-z, 1)
    O['sum_pt'] = tot; O['log_sum_pt'] = np.log(tot); O['mass'] = m; O['mass_over_sum_pt'] = m / tot; O['mass_over_sum_pt_sq'] = (m / tot) ** 2
    O['max_dr'] = np.where(real, dr, -np.inf).max(1); O['pt_dispersion'] = np.sqrt((z * z).sum(1))
    O['z_1st'], O['z_2nd'], O['z_3rd'] = zs[:, 0], zs[:, 1], zs[:, 2]; O['z_top5'] = zs[:, :5].sum(1)
    O['girth'] = (z * dr).sum(1); O['girth2'] = (z * dr ** 2).sum(1)
    O['mean_eta'] = (z * eta).sum(1); O['mean_eta2'] = (z * eta ** 2).sum(1); O['mean_phi'] = (z * phi).sum(1); O['mean_phi2'] = (z * phi ** 2).sum(1)
    O['n_particles'] = nreal.astype(float)
    for c in (1, 5, 10, 50): O[f'n_pt_above_{c}'] = (pt > c).sum(1).astype(float)
    tau_ax = {k: _kmeans(pt, eta, phi, k) for k in (1, 2, 3, 4)}
    tau = {k: (z * D.min(-1)).sum(1) / R0 for k, (_, D) in tau_ax.items()}
    O['tau21'] = tau[2] / np.maximum(tau[1], 1e-12); O['tau32'] = tau[3] / np.maximum(tau[2], 1e-12)
    # e2 / e3 (library 'e2', 'C2', 'D2'): the 24 hardest by pT (stable sort); the other ECFs: the first 24 / 12 slots
    h3 = min(n, H3); top = np.argsort(-pt, 1, kind='stable')[:, :h3]
    e2, _, e3, _, _, _, R, zt = _ecf(*(np.take_along_axis(a, top, 1) for a in (z, eta, phi)))
    O['e2'] = e2; O['C2'] = e3 / np.maximum(e2 ** 2, 1e-12); O['D2'] = e3 / np.maximum(e2 ** 3, 1e-12)
    e2, e2b2, e3, e3b2, g31, g32, R, zt = _ecf(z[:, :h3], eta[:, :h3], phi[:, :h3])
    iua = np.triu_indices(n, 1)
    O['e2_sq'] = (z[:, iua[0]] * z[:, iua[1]] * ((eta[:, iua[0]] - eta[:, iua[1]]) ** 2 + (phi[:, iua[0]] - phi[:, iua[1]]) ** 2)).sum(1)
    O['LHA'] = (z * np.sqrt(dr / 0.8)).sum(1); O['centroid_offset'] = np.hypot(O['mean_eta'], O['mean_phi'])
    ta, tb, tc = O['mean_eta2'], (z * eta * phi).sum(1), O['mean_phi2']
    disc = np.sqrt(np.maximum((ta - tc) ** 2 / 4 + tb ** 2, 0)); lam1, lam2 = (ta + tc) / 2 + disc, np.maximum((ta + tc) / 2 - disc, 0)
    O['planar_flow'] = 4 * (ta * tc - tb ** 2) / np.maximum((ta + tc) ** 2, 1e-12); O['eccentricity'] = 1 - lam2 / np.maximum(lam1, 1e-12)
    O['width'] = ta + tc; O['lam1'] = lam1; O['lam2'] = lam2; O['orientation_deg'] = np.degrees(0.5 * np.arctan2(2 * tb, ta - tc))
    # ---- the two / three hardest ----
    O['dr01'] = dist(0, 1); O['pt1_over_pt0'] = pt[:, 1] / np.maximum(pt[:, 0], 1e-9)
    O['pt_balance01'] = np.minimum(pt[:, 0], pt[:, 1]) / np.maximum(pt[:, 0] + pt[:, 1], 1e-9); O['pt1_dr01'] = pt[:, 1] * dist(0, 1)
    m01 = pm(0, 1)
    if n >= 3:                                   # (the networks here see 8 or 64 particles)
        m02, m12 = pm(0, 2), pm(1, 2); m012 = np.sqrt(m01 ** 2 + m02 ** 2 + m12 ** 2)
        O['m01'] = m01; O['m012'] = m012; O['min_pair_mass'] = np.minimum(np.minimum(m01, m02), m12); O['max_pair_mass'] = np.maximum(np.maximum(m01, m02), m12)
        r2 = real[:, 2]; a01 = dist(0, 1); a02 = np.where(r2, dist(0, 2), 0.0); a12 = np.where(r2, dist(1, 2), 0.0)
        O['dr02'] = a02; O['dr12'] = a12
        O['dr_min_012'] = np.where(r2, np.minimum(np.minimum(a01, a02), a12), a01); O['dr_max_012'] = np.where(r2, np.maximum(np.maximum(a01, a02), a12), a01)
        O['pt2_over_pt0'] = pt[:, 2] / np.maximum(pt[:, 0], 1e-9)
        O['mratio_min_012'] = O['min_pair_mass'] / np.maximum(m012, 1e-9); O['mratio_max_012'] = O['max_pair_mass'] / np.maximum(m012, 1e-9)
    # ---- rings, integrated jet shape, counting ----
    for lo, hi in RINGS:
        ring = real & (dr >= lo) & (dr < (hi or 10)); t = ring_tag(lo, hi)
        O[f'n_dr_{t}'] = ring.sum(1).astype(float); O[f'z_dr_{t}'] = (z * ring).sum(1)
    for r in (0.1, 0.2, 0.3): O[f'psi_{r:g}'.replace('.', 'p')] = (z * (real & (dr < r))).sum(1)
    cz = np.cumsum(z, 1)
    for f in (0.5, 0.9): O[f'n_for_{int(100 * f)}pct'] = np.minimum((cz < f).sum(1) + 1, nreal).astype(float)
    O['pt_entropy'] = np.where(real, -z * np.log(np.where(real, z, 1.0)), 0.0).sum(1)
    # ---- each of the hardest particles ----
    for i in range(min(n, KH)):
        r = real[:, i]
        O[f'pt_{i}'], O[f'eta_{i}'], O[f'phi_{i}'] = pt[:, i], eta[:, i], phi[:, i]
        O[f'dr_{i}'] = np.where(r, dr[:, i], 0.0); O[f'z_{i}'] = z[:, i]
        O[f'abseta_{i}'] = np.abs(eta[:, i]); O[f'absphi_{i}'] = np.abs(phi[:, i]); O[f'zdr_{i}'] = z[:, i] * dr[:, i]
        if i >= 2:
            O[f'dr0_{i}'] = np.where(r, dist(0, i), 0.0); O[f'dr1_{i}'] = np.where(r, dist(1, i), 0.0)
            O[f'ptdr0_{i}'] = np.where(r, pt[:, i] * dist(0, i), 0.0); O[f'pair_mass_0_{i}'] = pm(0, i)
    # ---- the softest real particles beyond the hardest KH ----
    if n > KH:
        for s in range(1, KS + 1):
            j = nreal - s; ok = j >= KH; jj = np.clip(j, 0, n - 1)
            g = lambda a: np.where(ok, a[rows, jj], 0.0)
            O[f'soft{s}_pt'] = g(pt); O[f'soft{s}_z'] = g(z); O[f'soft{s}_abseta'] = g(np.abs(eta)); O[f'soft{s}_absphi'] = g(np.abs(phi)); O[f'soft{s}_dr'] = g(dr)
            O[f'soft{s}_dr0'] = np.where(ok, np.hypot(eta[rows, jj] - eta[:, 0], phi[rows, jj] - phi[:, 0]), 0.0)
    # ---- the k hardest ----
    for k in TOPK:
        if k >= n: break
        spk = pt[:, :k].sum(1)
        O[f'mass_top{k}'] = mass_of(k); O[f'sum_pt_top{k}'] = spk; O[f'z_top{k}_slots'] = spk / tot
        O[f'girth2_top{k}'] = (pt[:, :k] * dr[:, :k] ** 2).sum(1) / np.maximum(spk, 1e-9); O[f'n_real_top{k}'] = real[:, :k].sum(1).astype(float)
    # ---- energy correlation functions ----
    h4 = min(n, H4); C4 = np.array(list(itertools.combinations(range(h4), 4))).T
    d6 = np.stack([R[:, C4[p], C4[q]] for p, q in itertools.combinations(range(4), 2)], -1); w4 = zt[:, C4[0]] * zt[:, C4[1]] * zt[:, C4[2]] * zt[:, C4[3]]
    s6 = np.sort(d6, -1); e4 = (w4 * np.prod(d6, -1)).sum(1); g41, g42 = (w4 * s6[..., 0]).sum(1), (w4 * s6[..., 0] * s6[..., 1]).sum(1)
    O['e3'] = e3; O['e4'] = e4
    O['C3'] = e4 * e2 / np.maximum(e3 ** 2, EPS); O['D3'] = e4 * e2 ** 3 / np.maximum(e3 ** 3, EPS)
    O['N2'] = g32 / np.maximum(e2 ** 2, EPS); O['N3'] = g42 / np.maximum(g31 ** 2, EPS)
    O['M2'] = g31 / np.maximum(e2, EPS); O['M3'] = g41 / np.maximum(g31, EPS)
    O['C2_b2'] = e3b2 / np.maximum(e2b2 ** 2, EPS); O['D2_b2'] = e3b2 / np.maximum(e2b2 ** 3, EPS)
    # ---- N-subjettiness and subjets ----
    tau2 = {k: (z * D.min(-1) ** 2).sum(1) / R0 ** 2 for k, (_, D) in tau_ax.items()}
    for k in (1, 2, 3, 4): O[f'tau{k}'] = tau[k]
    O['tau43'] = tau[4] / np.maximum(tau[3], 1e-12); O['tau21_b2'] = tau2[2] / np.maximum(tau2[1], 1e-12)
    for k in (2, 3):
        ax, D = tau_ax[k]; near = D.argmin(-1)
        S = np.stack([(z * real * (near == j)).sum(1) for j in range(k)], 1); V = np.stack([(P4 * (real & (near == j))[..., None]).sum(1) for j in range(k)], 1)
        o = np.argsort(-S, 1, kind='stable'); S = np.take_along_axis(S, o, 1); V = np.take_along_axis(V, o[..., None], 1); A = np.take_along_axis(ax, o[..., None], 1)
        msj = _m4(*np.moveaxis(V, -1, 0))
        drs = {(p, q): np.hypot(A[:, p, 0] - A[:, q, 0], A[:, p, 1] - A[:, q, 1]) for p, q in itertools.combinations(range(k), 2)}
        if k == 2:
            O['sj2_zsoft'] = S[:, 1]; O['sj2_dr'] = drs[0, 1]; O['sj2_mass1'] = msj[:, 0]; O['sj2_mass2'] = msj[:, 1]
        else:
            for p in range(3): O[f'sj3_z{p + 1}'] = S[:, p]; O[f'sj3_mass{p + 1}'] = msj[:, p]
            for (p, q), v in drs.items(): O[f'sj3_dr{p + 1}{q + 1}'] = v
            dd = np.stack(list(drs.values()), -1); O['sj3_dr_min'] = dd.min(-1); O['sj3_dr_max'] = dd.max(-1)
            mp = np.stack([_m4(*(V[:, p] + V[:, q]).T) for p, q in itertools.combinations(range(3), 2)], -1)
            O['sj3_pair_mass_min'] = mp.min(-1); O['sj3_pair_mass_max'] = mp.max(-1)
            O['sj3_pairmin_over_m'] = mp.min(-1) / np.maximum(m, 1e-9); O['sj3_pairmax_over_m'] = mp.max(-1) / np.maximum(m, 1e-9)
    # ---- soft drop ----
    hs = min(n, HSD); O['sd_mass'], O['sd_zg'], O['sd_rg'], O['sd_nremoved'] = _softdrop(pt[:, :hs], eta[:, :hs], phi[:, :hs])
    return O
