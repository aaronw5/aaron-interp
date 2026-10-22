"""H80 (loop D, round D17; new claim D31, exploratory; context D5, D8, D13, D30): is the complete N-subjettiness basis enough?

Question. If every N-subjettiness value of the Datta-Larkoski M-body basis is held fixed, does a tagger still change its score?
  tau_N^beta = sum_i pT_i min_k dR_{i,k}^beta / sum_i pT_i     (k = 1..N subjet axes; dR in (rapidity y, azimuth phi))
  B_M = {tau_N^beta : N = 1..M-1, beta = 1/2, 1, 2}: 3M - 3 numbers (Datta & Larkoski, JHEP 06 (2017) 073: B_M spans M-body
  phase space). Axes: exclusive-kT subjets (pairwise kT clustering d_ij = min(pT_i^2, pT_j^2) dR_ij^2, E-scheme recombination,
  undone to N subjets) of the UNEDITED jet; subjet membership is held during the solve (axes = E-scheme direction of the
  members, so they move with the particles), and after the edit the jet is re-clustered from scratch and every basis value
  recomputed as the acceptance check.
Edit (per jet, jets with N >= NMIN real particles). Variables per particle: a_i = log pT_i, eta_i, phi_i (particle masses fixed);
  normalised units u = (a_i / 0.03, eta_i / s, phi_i / s), s = 0.01 / sqrt(2). Constraints C_A:
    every arm   log m_J, log pT_J (jet invariant mass and transverse momentum of the constituent sum), y_J, phi_J (E-scheme axis);
                N unchanged (no particle added, removed, split or merged)
    NS_M        + log tau_N^beta for N = 1..M-1, beta in (1/2, 1, 2)   (M = 3, 4, 6: 6, 9, 15 basis values)
    FREE        nothing else (control: the basis moves)
  A Gaussian vector v (one per jet, shared by every arm and step size) is projected onto the null space of dC_A/du at the
  unedited jet, rescaled to ||v_A||^2 = 3 N (eps / 0.01)^2 (RMS per particle: 3 % x eps/0.01 in pT and eps in dR), and then a
  Gauss-Newton minimum-norm correction brings C_A back to its unedited value. Accepted if the solver residual < 1e-7 and, after
  re-clustering, every held basis value, m_J and pT_J are within 0.5 % and the axis within 0.002 of their unedited values.
  eps in EPS = (0.01, 0.005, 0.0025, 0.00125); the arms share v, so the comparison NS_M vs FREE is paired per jet.
Response. s = own-class log-odds (log p_y - log(1 - p_y), true class y); Delta s = s(edited) - s(unedited).
  kappa_M = mean |Delta s(NS_M)| / mean |Delta s(FREE)| (basis-sufficiency ratio; 0 = reads nothing beyond B_M + m + pT,
  1 = holding the basis does not reduce the response).
References (trained on disjoint real jets of the same dataset): GBDT_M and MLP_M on [log tau of B_M, log m, log pT],
  M = 2..7 (saturation curve; GBDT_6 on NS_6 is the solver check: it must give ~0); LR(pT^D) on pT^D = sqrt(sum pT_i^2)/sum pT_i
  (IRC-unsafe, must move). C2ST: GBDT separating edited (half A) from unedited (half B) jets on N, m, pT, B_7, pT^D, two
  minimum pair distances and the 16 hardest constituents (log pT_i / pT_J, dy_i, dphi_i about the jet axis).
Samples: Top Landscape test 2k top + 2k QCD (rng 8001), reference 5k + 5k (rng 8011); JetClass first test file per class, random
  rows, 1,000 each of Tbqq, Wqq, Zqq, ZJetsToNuNu (rng 8002), reference 2,000 per class (rng 8012); KMT Pythia QG test 2k q + 2k g
  (rng 8003), reference 5k + 5k (rng 8013); hls4ml jedi_core.sample(400, 8004) (5 classes), reference 3,000 per class (rng 8014).
Run
  ./part_job.sh d-h80-run 'python -m interp.h80_nsub_basis_nullspace --part run --ds top,jc,qg,hls'
  ./job.sh d-h80-jedi 1 4 'python -m interp.h80_nsub_basis_nullspace --part jedi'
  AI_PY=conda ./job.sh d-h80-stats 1 4 'python -m interp.h80_nsub_basis_nullspace --part stats && python -m ... --part fig'
Outputs: $AI_RESULTS/physics/h80_<ds>.npz / .json, h80_hls_tables.npz, h80_hls_jedi.npz, h80_stats.json, h80_nsub_basis.png
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
T0 = time.time()
EPS = (0.01, 0.005, 0.0025, 0.00125)
SA, SANG = 0.03, 0.01 / np.sqrt(2)
BETAS = np.array([0.5, 1.0, 2.0])
NMAXB = 6                                   # tau_N for N = 1..6 (B_7) computed for every jet
ARMS = dict(NS3=3, NS4=4, NS6=6, FREE=1)    # M (basis B_M held); FREE holds no basis value
NMIN = 13
TOL = 0.005
SEEDS = dict(top=8001, jc=8002, qg=8003, hls=8004)
RJ = dict(top=0.8, jc=0.8, qg=0.4, hls=0.8)
JC_CLS = ('QCD', 'Zqq', 'Wqq', 'Tbqq')      # ZJetsToNuNu, ZToQQ, WToQQ, TTBar (CL index 0, 6, 7, 8)
NB, NPERM = 1000, 10000
FLOOR_EPS = 1e-05      # FREE edit scaled to 1e-5 (RMS 0.003 % in pT, 1e-5 in dR): numerical floor of each model's response


def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)


try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    njit = lambda *a, **k: (lambda f: f); prange = range; HAVE_NUMBA = False


# ============================================================================================ exclusive-kT clustering + basis
@njit(cache=False)
def _dkt(a, b, pt2, yy, ph):
    dphi = abs(ph[a] - ph[b])
    if dphi > np.pi: dphi = 2 * np.pi - dphi
    return min(pt2[a], pt2[b]) * ((yy[a] - yy[b]) ** 2 + dphi ** 2)


@njit(cache=False)
def _one_jet(P, n, nmax, betas, tau, lab):
    cx = P[:n, 0].copy(); cy = P[:n, 1].copy(); cz = P[:n, 2].copy(); cE = P[:n, 3].copy()
    pt = np.sqrt(cx ** 2 + cy ** 2); yi = np.empty(n); phi = np.empty(n)
    for i in range(n):
        e = max(cE[i], abs(cz[i]) + 1e-12); yi[i] = 0.5 * np.log((e + cz[i]) / (e - cz[i])); phi[i] = np.arctan2(cy[i], cx[i])
    pt2 = pt ** 2; yy = yi.copy(); ph = phi.copy(); alive = np.ones(n, np.bool_); root = np.arange(n)
    nnd = np.full(n, np.inf); nnj = -np.ones(n, np.int64)
    for a in range(n):
        for b in range(n):
            if b != a:
                d = _dkt(a, b, pt2, yy, ph)
                if d < nnd[a]: nnd[a] = d; nnj[a] = b
    cnt = n
    for N in range(1, nmax + 1):
        if n <= N:
            for i in range(n): lab[N, i] = i
    while cnt > 1:
        a = -1; best = np.inf
        for i in range(n):
            if alive[i] and nnd[i] < best: best = nnd[i]; a = i
        b = nnj[a]
        cx[a] += cx[b]; cy[a] += cy[b]; cz[a] += cz[b]; cE[a] += cE[b]
        pt2[a] = cx[a] ** 2 + cy[a] ** 2; e = max(cE[a], abs(cz[a]) + 1e-12)
        yy[a] = 0.5 * np.log((e + cz[a]) / (e - cz[a])); ph[a] = np.arctan2(cy[a], cx[a])
        alive[b] = False; nnd[b] = np.inf
        for i in range(n):
            if root[i] == b: root[i] = a
        cnt -= 1
        nnd[a] = np.inf; nnj[a] = -1
        for c in range(n):
            if alive[c] and c != a:
                d = _dkt(a, c, pt2, yy, ph)
                if d < nnd[a]: nnd[a] = d; nnj[a] = c
        for c in range(n):
            if not alive[c] or c == a: continue
            if nnj[c] == a or nnj[c] == b:
                nnd[c] = np.inf; nnj[c] = -1
                for q in range(n):
                    if alive[q] and q != c:
                        d = _dkt(c, q, pt2, yy, ph)
                        if d < nnd[c]: nnd[c] = d; nnj[c] = q
            else:
                d = _dkt(c, a, pt2, yy, ph)
                if d < nnd[c]: nnd[c] = d; nnj[c] = a
        if cnt <= nmax:
            mp = -np.ones(n, np.int64); k = 0
            for i in range(n):
                r = root[i]
                if mp[r] < 0: mp[r] = k; k += 1
                lab[cnt, i] = mp[r]
    # basis values from the labels (axes = E-scheme direction of each subjet's members)
    S = pt.sum()
    for N in range(1, nmax + 1):
        if n <= N: continue
        AX = np.zeros((N, 4))
        for i in range(n):
            k = lab[N, i]; AX[k, 0] += P[i, 0]; AX[k, 1] += P[i, 1]; AX[k, 2] += P[i, 2]; AX[k, 3] += P[i, 3]
        Ay = np.empty(N); Ap = np.empty(N)
        for k in range(N):
            e = max(AX[k, 3], abs(AX[k, 2]) + 1e-12); Ay[k] = 0.5 * np.log((e + AX[k, 2]) / (e - AX[k, 2])); Ap[k] = np.arctan2(AX[k, 1], AX[k, 0])
        acc = np.zeros(betas.shape[0])
        for i in range(n):
            dm = np.inf
            for k in range(N):
                dphi = abs(phi[i] - Ap[k])
                if dphi > np.pi: dphi = 2 * np.pi - dphi
                d2 = (yi[i] - Ay[k]) ** 2 + dphi ** 2
                if d2 < dm: dm = d2
            for bi in range(betas.shape[0]): acc[bi] += pt[i] * (dm + 1e-14) ** (betas[bi] / 2)
        for bi in range(betas.shape[0]): tau[N - 1, bi] = acc[bi] / S


@njit(parallel=True, cache=False)
def basis_batch(P, nn, nmax, betas):
    J, L = P.shape[0], P.shape[1]
    tau = np.zeros((J, nmax, betas.shape[0])); lab = np.zeros((J, nmax + 1, L), np.int64)
    for j in prange(J):
        if nn[j] >= 1: _one_jet(P[j], nn[j], nmax, betas, tau[j], lab[j])
    return tau, lab


def table_P4(T):
    """real particles first in their table order (tables are left-packed); (J, L, 4) px, py, pz, E and counts"""
    real = T['pt'] > 0; pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']
    P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
    return np.ascontiguousarray(P), real.sum(1).astype(np.int64)


def obs(T, with_lab=False):
    """exact jet observables: N, m, pT_J, y_J, phi_J, pT^D, tau (J, 6, 3) (exclusive-kT axes, re-clustered), labels"""
    P, nn = table_P4(T); S = P.sum(1)
    real = T['pt'] > 0
    assert np.all(real[np.arange(real.shape[1])[None, :] < nn[:, None]]), 'table not left-packed'
    tau, lab = basis_batch(P, nn, NMAXB, BETAS)
    pt = T['pt']; ptd = np.sqrt((pt ** 2).sum(1)) / np.maximum(pt.sum(1), 1e-12)
    E, Z = S[:, 3], S[:, 2]
    O = dict(N=nn.astype(float), m=np.sqrt(np.maximum(E ** 2 - (S[:, :3] ** 2).sum(1), 0)), jpt=np.hypot(S[:, 0], S[:, 1]),
             jy=0.5 * np.log((E + Z) / (E - Z)), jphi=np.arctan2(S[:, 1], S[:, 0]), ptd=ptd, tau=tau)
    if with_lab: O['lab'] = lab
    return O


def basis_feats(O, M):
    """[log tau of B_M, log m, log pT]"""
    t = O['tau'][:, :M - 1, :].reshape(len(O['m']), -1)
    return np.c_[np.log(t + 1e-9), np.log(np.maximum(O['m'], 1e-3)), np.log(O['jpt'])]


# ============================================================================================================ solver (torch)
class Solver:
    def __init__(self, dev):
        import torch
        self.t = torch; self.dev = dev

    def pack(self, T, rows, lab):
        t = self.t; n = int((T['pt'][rows] > 0).sum(1).max()); g = lambda a: t.from_numpy(np.ascontiguousarray(a[rows, :n], np.float64)).to(self.dev)
        pt, eta, phi, m = g(T['pt']), g(T['eta']), g(T['phi']), g(T['m']); mk = pt > 0
        axp = t.from_numpy(T['ax_phi'][rows].astype(np.float64)).to(self.dev)
        f = axp[:, None] + self._wrap(phi - axp[:, None])
        u = t.cat([t.where(mk, t.log(pt.clamp_min(1e-30)) / SA, t.zeros_like(pt)), t.where(mk, eta / SANG, t.zeros_like(pt)),
                   t.where(mk, f / SANG, t.zeros_like(pt))], 1)
        # tie groups: particles at the same position (dR < 1e-5; e.g. Top Landscape has ~2 such pairs per jet) share their
        # angular coordinates, so the edit moves them together and never separates them
        de = eta[:, :, None] - eta[:, None, :]; dp = self._wrap(phi[:, :, None] - phi[:, None, :])
        same = (de ** 2 + dp ** 2 < 1e-10) & mk[:, :, None] & mk[:, None, :]
        A = same.double() / same.double().sum(2, keepdim=True).clamp_min(1)
        # tied particles move rigidly: coordinates = group mean + their original (sub-1e-5) offset from it
        off = t.cat([u[:, n:2 * n] - (A @ u[:, n:2 * n, None])[..., 0], u[:, 2 * n:] - (A @ u[:, 2 * n:, None])[..., 0]], 1)
        S = dict(n=n, mk=mk, m=m * mk, u=u, A=A, off=off, mk3=t.cat([mk, mk, mk], 1).double(), n_tied=(same.sum(2) > 1).sum(1))
        self.set_labels(S, lab[rows]); return S

    def set_labels(self, S, lab):
        t = self.t; n = S['n']; S['lab'] = np.ascontiguousarray(lab[:, :, :n]); S['oh'] = {}
        for N in range(2, NMAXB):
            L = t.from_numpy(S['lab'][:, N]).to(self.dev)
            S['oh'][N] = t.nn.functional.one_hot(L, N).double() * S['mk'][..., None].double()

    def recluster(self, S, u):
        pt, eta, phi = [z.detach().cpu().numpy() for z in self.coords(S, u)]; m = S['m'].cpu().numpy(); mk = S['mk'].cpu().numpy()
        P = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * mk[..., None]
        _, lab = basis_batch(np.ascontiguousarray(P), mk.sum(1).astype(np.int64), NMAXB, BETAS)
        return lab

    def _wrap(self, d):
        return self.t.remainder(d + np.pi, 2 * np.pi) - np.pi

    def coords(self, S, u):
        n = S['n']; mk = S['mk']
        A = S['A']; pt = self.t.exp(u[:, :n] * SA) * mk
        o = S['off']
        return pt, ((A @ u[:, n:2 * n, None])[..., 0] + o[:, :n]) * SANG, ((A @ u[:, 2 * n:, None])[..., 0] + o[:, n:]) * SANG

    def C(self, S, u, M):
        t = self.t; pt, eta, phi = self.coords(S, u); mk = S['mk']
        px, py, pz = pt * t.cos(phi), pt * t.sin(phi), pt * t.sinh(eta)
        E = t.sqrt((pt * t.cosh(eta)) ** 2 + S['m'] ** 2 + 1e-30) * mk
        X, Y, Z, EE = px.sum(1), py.sum(1), pz.sum(1), E.sum(1)
        yJ = 0.5 * t.log((EE + Z) / (EE - Z)); pJ = t.atan2(Y, X)
        out = [0.5 * t.log(EE ** 2 - X ** 2 - Y ** 2 - Z ** 2), 0.5 * t.log(X ** 2 + Y ** 2), yJ, pJ]
        if M > 1:
            Es = t.where(mk, E, t.ones_like(E)); pzs = t.where(mk, pz, t.zeros_like(pz)); yi = 0.5 * t.log((Es + pzs) / (Es - pzs))
            spt = pt.sum(1)
            for N in range(1, M):
                if N == 1: Yk, Pk = yJ[:, None], pJ[:, None]
                else:
                    oh = S['oh'][N]; ax = [t.einsum('bl,bln->bn', q, oh) for q in (px, py, pz, E)]
                    Yk = 0.5 * t.log((ax[3] + ax[2]) / (ax[3] - ax[2])); Pk = t.atan2(ax[1], ax[0])
                d2 = (yi[:, :, None] - Yk[:, None, :]) ** 2 + self._wrap(phi[:, :, None] - Pk[:, None, :]) ** 2
                dm = d2.min(-1).values
                for b in BETAS: out.append(t.log((pt * (dm + 1e-14) ** (b / 2)).sum(1) / spt))
        return t.stack(out, 1)

    def jac(self, S, u, M):
        t = self.t; u = u.detach().requires_grad_(True); c = self.C(S, u, M); K = c.shape[1]
        J = t.stack([t.autograd.grad(c[:, k].sum(), u, retain_graph=k < K - 1)[0] * S['mk3'] for k in range(K)], 1)
        return c.detach(), J

    def project(self, S, u0, v, M):
        t = self.t; _, J = self.jac(S, u0, M)
        A = J @ J.transpose(1, 2) + 1e-12 * t.eye(J.shape[1], dtype=J.dtype, device=J.device)
        return v - (J.transpose(1, 2) @ t.linalg.solve(A, J @ v[..., None]))[..., 0]

    def solve(self, S, u, target, M, iters=60, maxstep=3.0):
        t = self.t; u = u.clone()
        for _ in range(iters):
            c, J = self.jac(S, u, M); res = target - c; res[:, 3] = self._wrap(res[:, 3])
            if float(res.abs().max()) < 1e-11: break
            A = J @ J.transpose(1, 2) + 1e-14 * t.eye(J.shape[1], dtype=J.dtype, device=J.device)
            d = (J.transpose(1, 2) @ t.linalg.solve(A, res[..., None]))[..., 0]
            sc = (maxstep / d.abs().amax(1).clamp_min(1e-30)).clamp(max=1.0); u = u + d * sc[:, None]
        c = self.C(S, u, M).detach(); r = target - c; r[:, 3] = self._wrap(r[:, 3])
        return u, r.abs().amax(1)


def make_edits(T, O0, seed, dev, chunk=1000):
    """-> {cond: table}, {cond: info dict}; cond = f'{arm}@{eps}'"""
    import torch
    Sv = Solver(dev); J = len(T['pt']); rows_all = np.arange(J); E = {}; info = {}
    for a, e in [(a, e) for a in ARMS for e in EPS] + [('FLOOR', FLOOR_EPS)]:
        if True:
            c = f'{a}@{e}'; E[c] = dict(pt=T['pt'].copy(), eta=T['eta'].copy(), phi=T['phi'].copy(), m=T['m'], ax_eta=T['ax_eta'],
                                        ax_phi=T['ax_phi'], ext=T['ext'])
            info[c] = dict(res=np.zeros(J), rms_dlogpt=np.zeros(J), rms_dr=np.zeros(J), n_recluster=np.zeros(J), n_tied=np.zeros(J))
    for s in range(0, J, chunk):
        rows = rows_all[s:s + chunk]; S = Sv.pack(T, rows, O0['lab']); n = S['n']; u0 = S['u']
        rng = np.random.default_rng([seed, 80, s])
        v = torch.from_numpy(rng.standard_normal(u0.shape)).to(dev) * S['mk3']; nn = S['mk'].sum(1).double()
        A = S['A']; v = torch.cat([v[:, :n], (A @ v[:, n:2 * n, None])[..., 0], (A @ v[:, 2 * n:, None])[..., 0]], 1)
        lab0 = S['lab'].copy()
        for a, M in ARMS.items():
            Sv.set_labels(S, lab0); c0, _ = Sv.jac(S, u0, M); vA = Sv.project(S, u0, v, M)
            vA = vA * (torch.sqrt(3 * nn) / vA.norm(dim=1).clamp_min(1e-30))[:, None]
            for e in EPS + ((FLOOR_EPS,) if a == 'FREE' else ()):
                Sv.set_labels(S, lab0); u, r = Sv.solve(S, u0 + vA * (e / 0.01), c0, M); c = f'{"FLOOR" if e == FLOOR_EPS else a}@{e}'
                nre = np.zeros(len(rows), int)
                for _ in range(3):      # re-cluster the edited jet; where the subjet membership changed, solve again with it
                    if M <= 2: break
                    lab1 = Sv.recluster(S, u); ch = np.any(lab1[:, 2:M] != S['lab'][:, 2:M], axis=(1, 2))
                    if not ch.any(): break
                    nre += ch; Sv.set_labels(S, lab1); u, r = Sv.solve(S, u, c0, M)
                info[c]['n_recluster'][rows] = nre
                pt, eta, phi = [z.detach().cpu().numpy() for z in Sv.coords(S, u)]; mk = S['mk'].cpu().numpy()
                Ec = E[c]; sl = (rows, slice(0, n))
                Ec['pt'][sl] = np.where(mk, pt, 0); Ec['eta'][sl] = np.where(mk, eta, 0)
                Ec['phi'][sl] = np.where(mk, (phi + np.pi) % (2 * np.pi) - np.pi, 0)
                du = (u - u0).detach().cpu().numpy(); cnt = np.maximum(mk.sum(1), 1)
                info[c]['res'][rows] = r.cpu().numpy(); info[c]['n_tied'][rows] = S['n_tied'].cpu().numpy()
                info[c]['rms_dlogpt'][rows] = np.sqrt(((du[:, :n] * SA) ** 2).sum(1) / cnt)
                info[c]['rms_dr'][rows] = np.sqrt((((du[:, n:2 * n] * SANG) ** 2) + (du[:, 2 * n:] * SANG) ** 2).sum(1) / cnt)
        log(f'  edits {s + len(rows)}/{J}')
    return E, info


def accept(O0, O1, res, M):
    rel = lambda a, b: np.abs(a / b - 1)
    # axis shift scaled so that 0.002 in y or phi counts as the 0.5 % tolerance
    dv = [rel(O1['m'], O0['m']), rel(O1['jpt'], O0['jpt']), np.abs(O1['jy'] - O0['jy']) * (TOL / 0.002),
          np.abs((O1['jphi'] - O0['jphi'] + np.pi) % (2 * np.pi) - np.pi) * (TOL / 0.002)]
    if M > 1: dv.append(rel(O1['tau'][:, :M - 1], O0['tau'][:, :M - 1]).reshape(len(res), -1).max(1))
    dev = np.max(np.stack(dv, 1), 1)
    return (res < 1e-7) & (dev < TOL) & (O1['N'] == O0['N']), dev


# =========================================================================================================== references
def train_refs(Op, yp, classes_used):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    R = {}
    for M in range(2, NMAXB + 2):
        X = basis_feats(Op, M)
        R[f'GBDT_B{M}'] = (HistGradientBoostingClassifier(max_iter=400, early_stopping=True, random_state=0).fit(X, yp), M)
        R[f'MLP_B{M}'] = (make_pipeline(StandardScaler(), MLPClassifier((64, 64), max_iter=500, early_stopping=True, random_state=0)).fit(X, yp), M)
    R['LR_pTD'] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Op['ptd'][:, None], yp), 0)
    log(f'references trained on {len(yp)} jets')
    return R


def ref_logits(R, O, C):
    out = {}
    for k, (clf, M) in R.items():
        X = Op_X(O, M); p = np.clip(clf.predict_proba(X), 1e-12, 1); L = np.full((len(X), C), np.log(1e-12))
        L[:, clf.classes_.astype(int)] = np.log(p); out[k] = L
    return out


def Op_X(O, M):
    return O['ptd'][:, None] if M == 0 else basis_feats(O, M)


# ================================================================================================================ C2ST
def c2st_feats(T, O):
    P, nn = table_P4(T); J, L = P.shape[:2]; pt = T['pt']; real = pt > 0
    E = np.where(real, P[..., 3], 1); pz = np.where(real, P[..., 2], 0); yi = 0.5 * np.log((E + pz) / np.maximum(E - pz, 1e-12))
    dy = np.where(real, yi - O['jy'][:, None], 0); dp = np.where(real, (T['phi'] - O['jphi'][:, None] + np.pi) % (2 * np.pi) - np.pi, 0)
    o = np.argsort(-pt, 1, kind='stable'); tk = lambda a: np.take_along_axis(a, o, 1)
    pts, dys, dps = tk(pt), tk(dy), tk(dp); K = 16
    pad = lambda a: a[:, :K] if a.shape[1] >= K else np.pad(a, ((0, 0), (0, K - a.shape[1])))
    mind, mind6 = np.zeros(J), np.zeros(J)
    for s in range(0, J, 500):
        sl = slice(s, s + 500); r = tk(real)[sl]
        D = np.hypot(dys[sl, :, None] - dys[sl, None, :], (dps[sl, :, None] - dps[sl, None, :] + np.pi) % (2 * np.pi) - np.pi)
        bad = ~(r[:, :, None] & r[:, None, :]) | np.eye(L, dtype=bool)[None]
        D = np.where(bad, 9.0, D); mind[sl] = D.reshape(len(D), -1).min(1); mind6[sl] = D[:, :6, :6].reshape(len(D), -1).min(1)
    return np.c_[O['N'], np.log(np.maximum(O['m'], 1e-3)), np.log(O['jpt']), np.log(O['tau'].reshape(J, -1) + 1e-9), O['ptd'],
                 np.log(mind + 1e-9), np.log(mind6 + 1e-9), np.log(np.maximum(pad(pts), 1e-9) / O['jpt'][:, None]), pad(dys), pad(dps)]


def c2st(F0, F1, seed=0, ok=None):
    """edited jets (half A, accepted jets only) vs unedited jets (half B, disjoint jets)"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); J = len(F0); p = rng.permutation(J); A, B = p[: J // 2], p[J // 2:]
    if ok is not None:
        A = A[ok[A]]; B = B[ok[B]]; k = min(len(A), len(B)); A, B = A[:k], B[:k]   # both halves from accepted jets (no selection effect)
    X = np.concatenate([F1[A], F0[B]]); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    sp = rng.permutation(len(y)); tr, te = sp[: len(y) // 2], sp[len(y) // 2:]
    pr = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], pr); bs = []
    for _ in range(200):
        ix = rng.integers(0, len(te), len(te)); bs.append(roc_auc_score(y[te][ix], pr[ix]))
    return dict(auc=float(auc), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


# ============================================================================================================ datasets
def H71():
    from interp import h71_energy_scale_panel as M
    return M


def keepN(T, y, nmin):
    n = (T['pt'] > 0).sum(1); k = n >= nmin; sub = {kk: (v[k] if isinstance(v, np.ndarray) else v) for kk, v in T.items() if kk != 'ext'}
    sub['ext'] = {kk: v[k] for kk, v in T['ext'].items()}; return sub, y[k], k


def load_top(dev):
    import torch
    from interp import data_top as dt
    from interp.h4_top_wmass import WeaverNet, LNet, kin3_of
    from interp.h12_boost_panel import table_from_full
    M71 = H71(); seed = SEEDS['top']
    d = dt.DST / 'test'; ylab = np.load(d / 'label.npy')
    ev = M71.H._rows(ylab, 2000, seed); used = np.zeros(len(ylab), bool); used[ev] = True; r = np.random.default_rng(seed + 10)
    rf = np.sort(np.concatenate([r.choice(np.flatnonzero((ylab == c) & ~used), 5000, replace=False) for c in (0, 1)]))
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy')
    tab = lambda ii: table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1]))
    T0, Tp = tab(ev), tab(rf); y, yp = ylab[ev].astype(int), ylab[rf].astype(int)

    def prep(T):
        pt, eta, phi, m = T['pt'], T['eta'], T['phi'], T['m']; real = pt > 0
        P4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), np.sqrt((pt * np.cosh(eta)) ** 2 + m ** 2)], -1) * real[..., None]
        return dict(P4=P4.astype(np.float32), nn=real.sum(1))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(ParT_kin='0.937 (H64) / 0.940 (H2)', PN_kin='0.940 (H64)', LorentzNet='0.9433 (H64)', PFN='0.9293 (H64)', EFN='0.9227 (H64)')
    models = {}; wf = lambda net: (lambda X: two(net.score(X['P4'], X['nn'])))
    top = V / 'models' / 'top'
    def wck(dn):
        d_ = top / dn
        if dn.endswith('_shuf') and (d_ / 'net_epoch-19_state.pt').exists(): return d_ / 'net_epoch-19_state.pt'
        return d_ / 'net_best_epoch_state.pt'
    for name, kind, dn, role in (('ParT_kin', 'ParT', 'part_kin', 'trained'), ('PN_kin', 'PN', 'pnet_kin', 'trained'),
                                 ('PN_kin_s1', 'PN', 'pnet_kin_s1', 'trained'), ('PN_kin_shuf', 'PN', 'pnet_kin_shuf', 'shuf')):
        if not wck(dn).exists(): log(f'top {name}: missing {wck(dn)}'); continue
        net = WeaverNet(kind, wck(dn), dev); models[name] = dict(f=wf(net), role=role, known=known.get(name), arch=kind, ckpt=str(wck(dn)))
        if name in ('ParT_kin', 'PN_kin'):
            n2 = copy.copy(net); n2.model = M71.null_torch(net.model, 7); models[f'{name}_perm'] = dict(f=wf(n2), role='perm', arch=kind)
    ln = LNet(dev, mass64=True)
    models['LorentzNet'] = dict(f=(lambda X: two(ln.score(X['P4'], X['nn'], np.random.default_rng(seed + 5).permutation(len(X['nn']))))),
                                role='trained', known=known['LorentzNet'], arch='LorentzNet')
    from interp.h64_axis_boost_panel import null_flow

    class Flow2:
        def __init__(self, d_, kind):
            self.arch = kind; W = np.load(d_ / 'weights.npz'); keys = sorted(W.files)
            layers = sorted({k.rsplit('_', 1)[0] for k in keys}, key=lambda s: int(s.split('_')[0]))
            Ls = [(W[f'{l}_0'].astype(np.float32), W[f'{l}_1'].astype(np.float32)) for l in layers]
            self.phi, self.F, self.out = Ls[:3], Ls[3:6], Ls[6]
        from interp.h4_top_wmass import Flow as _F
        score = _F.score
    for name, dn, kind, role in (('PFN', 'pfn', 'pfn', 'trained'), ('PFN_s1', 'pfn_s1', 'pfn', 'trained'), ('EFN', 'efn', 'efn', 'trained'),
                                 ('PFN_shuf', 'pfn_shuf', 'pfn', 'shuf'), ('EFN_shuf', 'efn_shuf', 'efn', 'shuf')):
        if not (top / dn / 'weights.npz').exists(): log(f'top {name}: missing'); continue
        fl = Flow2(top / dn, kind); ff = lambda f_: (lambda X: two(f_.score(kin3_of(X['P4'], X['nn']))))
        models[name] = dict(f=ff(fl), role=role, known=known.get(name), arch=name.split('_')[0])
        if name in ('PFN', 'EFN'): models[f'{name}_perm'] = dict(f=ff(null_flow(fl, 'perm', 7)), role='perm', arch=name)
    return dict(T0=T0, y=y, Tp=Tp, yp=yp, classes=['QCD', 'top'], prep=prep, models=models, sig=[1], bkg=[0],
                sample=f'Top Landscape test, 2,000 top + 2,000 QCD random rows (rng {seed}); reference 5,000 + 5,000 disjoint rows (rng {seed + 10})')


def jc_rows(per, seed):
    import uproot
    from interp.part_core import read_root, fof, cat, sub
    rng = np.random.default_rng(seed); Js = []
    for c in JC_CLS:
        f = fof(c); n = uproot.open(f)['tree'].num_entries; idx = np.sort(rng.choice(n, per, replace=False)); parts = []
        for a in range(0, n, 20000):
            loc = idx[(idx >= a) & (idx < a + 20000)] - a
            if len(loc): parts.append(sub(read_root(f, start=a, stop=min(a + 20000, n)), loc))
        Js.append(cat(parts)); log(f'JetClass {c}: {per} random rows of {n} in {Path(f).name}')
    return cat(Js)


def load_jc(dev):
    from interp.h1_irc_panel import jc_table, pn_logits
    from interp.part_core import ParTNetwork, CL, inputs, build, sub
    from interp.p1v_verify_part import fwd
    from interp.pn_net import load_particlenet
    from jetdistill.config import MODELS
    import torch
    M71 = H71(); seed = SEEDS['jc']
    Jall = jc_rows(3000, seed); yall = Jall['label'].astype(int); rr = np.random.default_rng(seed + 10); ev, rf = [], []
    for c in np.unique(yall):
        ii = rr.permutation(np.flatnonzero(yall == c)); ev.append(ii[:1000]); rf.append(ii[1000:])
    ev, rf = np.sort(np.concatenate(ev)), np.sort(np.concatenate(rf))
    tab = lambda Jx: (lambda Tj: dict(pt=Tj['pt'], eta=Tj['eta'], phi=Tj['phi'], m=Tj['m'], ax_eta=Tj['ax_eta'], ax_phi=Tj['ax_phi'], ext=Tj['ext']))(jc_table(Jx))
    J0 = sub(Jall, ev); y = yall[ev]; T0 = tab(J0); Jr = sub(Jall, rf); yp = yall[rf]; Tp = tab(Jr); Tp['ext'] = {}; del Jall, Jr
    state = dict(J0=J0)

    def prep(T):
        Jx = dict(state['J0']); Jx['jet_eta'] = T['ax_eta'].astype(np.float32); Jx['jet_phi'] = T['ax_phi'].astype(np.float32)
        sg = np.where(Jx['jet_eta'] < 0, -1.0, 1.0)[:, None]; de, dp = M71.H.rel(T)
        return build(Jx, T['pt'], sg * de, dp, T['m'], extra=T['ext'])
    known = dict(ParT_full='0.8583 (H64, 10 classes)', ParT_kin='0.7416 (H64, 10 classes)', PN_full='0.8384 (H64)', PN_kin='0.7040 (H64)')
    pt_f = lambda net, n: (lambda J: fwd(net, *inputs(J, n))[1].astype(np.float64))
    models = {}
    for n in ('full', 'kin'):
        net = ParTNetwork(n); models[f'ParT_{n}'] = dict(f=pt_f(net, n), role='trained', known=known[f'ParT_{n}'], arch='ParT')
        nn_ = copy.copy(net); nn_.model = M71.null_torch(net.model, 7); models[f'ParT_{n}_perm'] = dict(f=pt_f(nn_, n), role='perm', arch='ParT')
    for s in (0, 1, 2):
        p = Path(f'/j-jepa-vol/aaron-interp/models/part_seeds/kin_s{s}/net_best_epoch_state.pt')
        if not p.exists(): log('missing', p); continue
        net = ParTNetwork('kin'); sd = torch.load(p, map_location='cpu'); sd = {k[4:] if k.startswith('mod.') else k: v for k, v in sd.items()}
        net.model.load_state_dict(sd, strict=True); net.model.eval()
        models[f'ParT_kin_s{s}'] = dict(f=pt_f(net, 'kin'), role='trained', known='P10 retrain (2M jets)', arch='ParT', ckpt=str(p))
    for n, dims in (('full', 17), ('kin', 7)):
        pn = load_particlenet(Path(MODELS).parent / 'particlenet' / f'ParticleNet_{n}.pt', dims, dev)
        mk = lambda p_, n=n: (lambda J: pn_logits(p_, J, n))
        models[f'PN_{n}'] = dict(f=mk(pn), role='trained', known=known[f'PN_{n}'], arch='PN')
        models[f'PN_{n}_perm'] = dict(f=mk(M71.null_torch(pn, 7)), role='perm', arch='PN')
    return dict(T0=T0, y=y, Tp=Tp, yp=yp, classes=list(CL), prep=prep, models=models, sig=[8, 7, 6], bkg=[0], state=state,
                sample=f'JetClass first test file per class (TTBar/Tbqq, WToQQ, ZToQQ, ZJetsToNuNu), 3,000 random rows per class (rng {seed}); '
                       f'1,000 per class evaluated, 2,000 per class reference (rng {seed + 10})')


def ef_from_npz(path, kind):
    import torch
    from interp import h5_qg_counting as H5
    W = np.load(path); ks = sorted(W.files)
    kern = [W[k] for k in ks if k.endswith('_0')]; bias = [W[k] for k in ks if k.endswith('_1')]
    net = H5.make_ef(kind); lin = [m for m in list(net.phi) + list(net.F) if isinstance(m, torch.nn.Linear)]
    assert len(lin) == len(kern) == 7, (len(lin), len(kern))
    with torch.no_grad():
        for i, (l, k, b) in enumerate(zip(lin, kern, bias)):
            k = k.copy()
            if i == 0: k[-2:] *= 0.4
            l.weight.copy_(torch.from_numpy(k.T)); l.bias.copy_(torch.from_numpy(b))
    return net


def load_qg(dev):
    from interp import data_qg as dq
    from interp import h5_qg_counting as Q
    from interp.h12_boost_panel import table_from_full
    M71 = H71(); seed = SEEDS['qg']
    d = dq.DST / 'test'; ylab = np.load(d / 'label.npy')
    ev = M71.H._rows(ylab, 2000, seed); used = np.zeros(len(ylab), bool); used[ev] = True; r = np.random.default_rng(seed + 10)
    rf = np.sort(np.concatenate([r.choice(np.flatnonzero((ylab == c) & ~used), 5000, replace=False) for c in (0, 1)]))
    Fm = np.load(d / 'full.npy', mmap_mode='r'); jet = np.load(d / 'jet.npy'); pidm = np.load(d / 'pid.npy', mmap_mode='r')

    def tab(ii):
        T = table_from_full(np.asarray(Fm[ii]), np.minimum(jet[ii, 5].astype(np.int64), Fm.shape[1])); T['m'] = np.zeros_like(T['m'])
        T['ext'] = dict(pid=np.asarray(pidm[ii]).astype(np.int64) * (T['pt'] > 0)); return T
    T0, Tp = tab(ev), tab(rf); Tp['ext'] = {}; y, yp = ylab[ev].astype(int), ylab[rf].astype(int)

    def prep(T):
        real = T['pt'] > 0
        return dict(pt=T['pt'], eta=T['eta'], phi=T['phi'], pid=T['ext']['pid'].astype(np.int64) * real, n=real.sum(1))
    M = Q.Models(dev)
    for name, arch in (('part_kin', 'part'), ('part_kin_s1', 'part'), ('pnet_kin', 'pn'), ('pnet_kin_shuf', 'pn')):
        try: M.add_weaver(name, arch, False)
        except Exception as ex: log(f'QG {name} not loaded: {ex!r}')
    for src in ('part_kin', 'pnet_kin'):
        if src in M.m: M.add_permuted(f'{src}_perm', src)
    try: M.add_lnet()
    except Exception as ex: log(f'QG LorentzNet not loaded: {ex!r}')
    for kind, dn in (('pfn', 'pfn'), ('efn', 'efn'), ('pfn', 'pfn_shuf'), ('efn', 'efn_shuf')):
        p = V / 'models' / 'qg' / dn / 'weights.npz'
        if p.exists(): M.add_torch(dn, ef_from_npz(p, kind), dict(weights=str(p)))
        else: log(f'QG {dn}: missing')
    if 'pfn' in M.m:
        import torch
        net2 = M71.null_torch(M.m['pfn'][3], 7); M.add_torch('pfn_perm', net2, dict(note='pfn permuted'))
    two = lambda s: np.stack([np.zeros_like(s), s], 1)
    known = dict(part_kin='0.841 (H64) / 0.827 (H76, AUC 0.907)', pnet_kin='0.8355 (H64)', lorentznet='0.8542 (H64)')
    role = lambda k: 'perm' if k.endswith('_perm') else ('shuf' if k.endswith('_shuf') else 'trained')
    arch = lambda k: dict(part='ParT', pnet='PN', pfn='PFN', efn='EFN', lorentznet='LorentzNet')[k.split('_')[0]]
    models = {k: dict(f=(lambda X, k=k: two(-M.logit(k, X))), role=role(k), known=known.get(k), arch=arch(k)) for k in M.m}
    return dict(T0=T0, y=y, Tp=Tp, yp=yp, classes=['gluon', 'quark'], prep=prep, models=models, sig=[1], bkg=[0], info=M.info,
                sample=f'KMT Pythia QG test, 2,000 quark + 2,000 gluon random rows (rng {seed}); reference 5,000 + 5,000 disjoint rows (rng {seed + 10})')


def load_hls(dev):
    from interp.h1_irc_panel import hls_table
    seed = SEEDS['hls']; d = np.load(V / 'data' / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r'); yall = np.asarray(d['y'])
    rng = np.random.default_rng(seed)    # identical draw to interp.jedi_core.sample(per_class=400, seed=8004)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yall == c), 400, replace=False) for c in range(5)]))
    r2 = np.random.default_rng(seed + 10); used = np.zeros(len(yall), bool); used[idx] = True
    ref = np.sort(np.concatenate([r2.choice(np.flatnonzero((yall == c) & ~used), 3000, replace=False) for c in range(5)]))
    x = d['x']; mk = lambda ii: {k: v for k, v in hls_table(np.asarray(x[ii])[..., :3]).items() if k != 'sgn'}
    T0, Tp = mk(idx), mk(ref)
    return dict(T0=T0, y=yall[idx].astype(int), Tp=Tp, yp=yall[ref].astype(int), classes=['g', 'q', 'W', 'Z', 't'], prep=None, models={},
                sig=[2, 3, 4], bkg=[0, 1], idx=idx, sample=f'hls4ml jedi_core.sample(per_class=400, seed={seed}); reference 3,000 other rows per class (rng {seed + 10})')


LOADERS = dict(top=load_top, jc=load_jc, qg=load_qg, hls=load_hls)


# ================================================================================================================== run
def sig_auc(lg, y, sig, bkg):
    """mean over signal classes k of AUC(log p_k - log sum_b p_b), class-k jets vs background jets"""
    from sklearn.metrics import roc_auc_score
    ls = lg - np.logaddexp.reduce(lg, 1, keepdims=True); lb = np.logaddexp.reduce(ls[:, bkg], 1); out = {}
    for k in sig:
        r = np.isin(y, [k] + list(bkg)); s = ls[r, k] - lb[r]
        out[k] = float(roc_auc_score(y[r] == k, s)) if np.ptp(s) > 0 else 0.5
    return float(np.mean(list(out.values()))), out


def run_ds(ds, dev, smoke):
    M71 = H71(); t0 = time.time(); D = LOADERS[ds](dev)
    T0, y, Tp, yp, C = D['T0'], D['y'], D['Tp'], D['yp'], len(D['classes'])
    n_in = len(y); T0, y, keep = keepN(T0, y, NMIN)
    if ds == 'jc': D['state']['J0'] = {k: (v[keep] if np.ndim(v) else v) for k, v in D['state']['J0'].items()}
    if smoke:
        r = np.sort(np.random.default_rng(0).choice(len(y), 200, replace=False))
        T0 = dict({k: (v[r] if isinstance(v, np.ndarray) else v) for k, v in T0.items() if k != 'ext'}, ext={k: v[r] for k, v in T0['ext'].items()})
        y = y[r]; keep = np.flatnonzero(keep)[r]
        if ds == 'jc': D['state']['J0'] = {k: (v[r] if np.ndim(v) else v) for k, v in D['state']['J0'].items()}
        rp = np.sort(np.random.default_rng(1).choice(len(yp), 3000, replace=False))
        Tp, yp = {k: (v[rp] if isinstance(v, np.ndarray) else v) for k, v in Tp.items()}, yp[rp]
    J = len(y); log(f'{ds}: {J} evaluated jets (N >= {NMIN}: kept {J}/{n_in}), {len(yp)} reference jets; numba {HAVE_NUMBA}')
    res = dict(ds=ds, sample=D['sample'], n_eval=J, n_before_cut=n_in, labels=np.bincount(y, minlength=C).tolist(), classes=D['classes'],
               eps=list(EPS), arms=ARMS, nmin=NMIN, tol=TOL, models={}, edits={}, c2st={}, saturation={}, info=D.get('info'))
    NPZ = dict(y=y.astype(np.int16))
    O0 = obs(T0, with_lab=True); Op = obs(Tp); log(f'{ds}: basis of {J} + {len(yp)} jets ({time.time() - t0:.0f} s)')
    R = train_refs(Op, yp, None)
    # unedited scores
    models = D['models']; X0 = D['prep'](M71.H.sortT(T0)) if D['prep'] else None
    for mn, Mo in models.items():
        lg = np.asarray(Mo['f'](X0), np.float64); lo = M71.logodds_all(lg); c = lg.argmax(1)
        rr = dict(role=Mo['role'], arch=Mo.get('arch'), acc=float((c == y).mean()), known=Mo.get('known'), ckpt=Mo.get('ckpt'),
                  per_class={D['classes'][k]: float((c[y == k] == k).mean()) for k in np.unique(y)}, logit_std=float(lg.std(0).mean()))
        rr['auc_sig'], rr['auc_by_class'] = sig_auc(lg, y, D['sig'], D['bkg'])
        res['models'][mn] = rr; NPZ[f'lo|{mn}|base'] = lo[np.arange(J), y].astype(np.float32); NPZ[f'lg|{mn}|base'] = lg.astype(np.float32)
        log(f'{ds} {mn} [{Mo["role"]}]: acc {rr["acc"]:.4f} (known {Mo.get("known")}) signal AUC {rr["auc_sig"]:.4f}')
    RL0 = ref_logits(R, O0, C)
    for k, lg in RL0.items():
        c = lg.argmax(1); a, ab = sig_auc(lg, y, D['sig'], D['bkg'])
        res['models'][k] = dict(role='ref', acc=float((c == y).mean()), auc_sig=a, auc_by_class=ab)
        NPZ[f'lo|{k}|base'] = M71.logodds_all(lg)[np.arange(J), y].astype(np.float32); NPZ[f'lg|{k}|base'] = lg.astype(np.float32)
    res['saturation'] = {k: dict(acc=v['acc'], auc_sig=v['auc_sig']) for k, v in res['models'].items() if v['role'] == 'ref'}
    log(f'{ds} saturation: ' + ' '.join(f'{k} {v["auc_sig"]:.4f}' for k, v in res['saturation'].items()))
    for k in ('N', 'm', 'jpt', 'ptd'): NPZ[f'obs|base|{k}'] = O0[k].astype(np.float32)
    NPZ['obs|base|tau'] = O0['tau'].astype(np.float32)
    # edits
    t1 = time.time(); E, info = make_edits(T0, O0, SEEDS[ds], dev); log(f'{ds}: edits done ({time.time() - t1:.0f} s)')
    F0 = c2st_feats(T0, O0); res['c2st']['none'] = c2st(F0, F0)
    hls_save = {}
    for cond, U in E.items():
        a, e = cond.split('@'); Mh = ARMS.get(a, 1); O1 = obs(U)
        ok, dv = accept(O0, O1, info[cond]['res'], Mh)
        lt = lambda q: np.log(q + 1e-12)
        dtau = np.abs(lt(O1['tau']) - lt(O0['tau']))           # (J, 6, 3)
        ed = dict(ok=float(ok.mean()), solver_ok=float((info[cond]['res'] < 1e-7).mean()), dev_median=float(np.median(dv)),
                  rms_dlogpt_median=float(np.median(info[cond]['rms_dlogpt'][ok])) if ok.any() else None,
                  rms_dr_median=float(np.median(info[cond]['rms_dr'][ok])) if ok.any() else None,
                  mean_abs_dlogtau_by_N=[float(dtau[ok, N].mean()) for N in range(NMAXB)] if ok.any() else None,
                  resolve_share=float((info[cond]['n_recluster'] > 0).mean()), tied_particles_mean=float(info[cond]['n_tied'].mean()),
                  mean_abs_dlog_ptd=float(np.abs(np.log(O1['ptd'][ok] / O0['ptd'][ok])).mean()) if ok.any() else None)
        res['edits'][cond] = ed
        NPZ[f'ok|{cond}'] = ok; NPZ[f'dev|{cond}'] = dv.astype(np.float32)
        NPZ[f'obs|{cond}|tau'] = O1['tau'].astype(np.float32); NPZ[f'obs|{cond}|ptd'] = O1['ptd'].astype(np.float32)
        for k in ('rms_dlogpt', 'rms_dr'): NPZ[f'{k}|{cond}'] = info[cond][k].astype(np.float32)
        if models:
            X1 = D['prep'](M71.H.sortT(U))
            for mn, Mo in models.items():
                NPZ[f'lo|{mn}|{cond}'] = M71.logodds_all(np.asarray(Mo['f'](X1), np.float64))[np.arange(J), y].astype(np.float32)
        for k, lg in ref_logits(R, O1, C).items(): NPZ[f'lo|{k}|{cond}'] = M71.logodds_all(lg)[np.arange(J), y].astype(np.float32)
        F1 = c2st_feats(U, O1); res['c2st'][cond] = dict(c2st(F0, F1, seed=1, ok=ok), n_edited=int(ok.sum() // 2))
        if cond in (f'FREE@{EPS[0]}', f'FREE@{EPS[-1]}', f'NS6@{EPS[-1]}', f'FLOOR@{FLOOR_EPS}'):
            # which feature block carries the separation: jet-level (N, m, pT, 18 tau, pT^D) / two minimum pair distances /
            # 16 hardest constituents (log pT_i/pT_J, dy, dphi)
            bl = dict(jet=slice(0, 22), pairs=slice(22, 24), constituents=slice(24, None))
            res.setdefault('c2st_blocks', {})[cond] = {k: c2st(F0[:, v], F1[:, v], seed=1, ok=ok)['auc'] for k, v in bl.items()}
            log(f'{ds} {cond} C2ST by block', res['c2st_blocks'][cond])
        if ds == 'hls': hls_save.update({f'{cond}|{k}': U[k].astype(np.float32) for k in ('pt', 'eta', 'phi')})
        mv = lambda mn: np.mean(np.abs(NPZ[f'lo|{mn}|{cond}'] - NPZ[f'lo|{mn}|base'])[ok]) if ok.any() else float('nan')
        log(f'{ds} {cond}: ok {ed["ok"]:.3f} (solver {ed["solver_ok"]:.3f}) rms dlogpT {ed["rms_dlogpt_median"]} rms dR {ed["rms_dr_median"]} '
            f'C2ST {res["c2st"][cond]["auc"]:.3f} | |ds| ' + ' '.join(f'{mn} {mv(mn):.4f}' for mn in list(models) + ['GBDT_B6', 'GBDT_B3', 'LR_pTD']))
    if ds == 'hls':
        np.savez_compressed(RES / f'h80_hls_tables{"_smoke" if smoke else ""}.npz', y=y, idx=D['idx'][keep],
                            base_pt=T0['pt'].astype(np.float32), base_eta=T0['eta'].astype(np.float32), base_phi=T0['phi'].astype(np.float32), **hls_save)
    if ds == 'top':   # example jet for the figure: a top jet accepted in NS6@0.01
        j = int(np.flatnonzero((y == 1) & NPZ['ok|NS6@0.01'])[0]); U = E['NS6@0.01']; n = int(O0['N'][j])
        np.savez_compressed(RES / f'h80_example_top{"_smoke" if smoke else ""}.npz', base=np.c_[T0['pt'][j, :n], T0['eta'][j, :n], T0['phi'][j, :n]],
                            edit=np.c_[U['pt'][j, :n], U['eta'][j, :n], U['phi'][j, :n]], free=np.c_[E['FREE@0.01']['pt'][j, :n], E['FREE@0.01']['eta'][j, :n], E['FREE@0.01']['phi'][j, :n]],
                            tau0=O0['tau'][j], m0=O0['m'][j], jpt0=O0['jpt'][j], tau1=obs({k: (v[j:j + 1] if isinstance(v, np.ndarray) else v) for k, v in U.items()})['tau'][0])
    tag = '_smoke' if smoke else ''
    np.savez_compressed(RES / f'h80_{ds}{tag}.npz', **NPZ)
    (RES / f'h80_{ds}{tag}.json').write_text(json.dumps(res, indent=1, default=str)); log(f'{ds}: wrote h80_{ds}{tag} ({time.time() - t0:.0f} s)')


def part_run(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True); log('device', dev)
    for ds in a.ds.split(','): run_ds(ds, dev, a.smoke)


def part_jedi(a):
    """JEDI-linear n64 3-feature (and a weight-permuted copy) on the saved hls4ml tables (jax env)"""
    from interp.jedi_core import Jedi
    from jetdistill import network as N
    from interp import h12_boost_panel as H
    tag = '_smoke' if a.smoke else ''; Z = dict(np.load(RES / f'h80_hls_tables{tag}.npz')); y = Z['y']; J = len(y)
    nets = dict(JEDI=Jedi(64, '3-feature')); orig = N.redraw_untrained; N.redraw_untrained = H.permute_weights
    try: nets['JEDI_perm'] = Jedi(64, '3-feature', untrained=True, seed=7)
    finally: N.redraw_untrained = orig
    out = {}
    conds = ['base'] + sorted({k.split('|')[0] for k in Z if '@' in k})
    for c in conds:
        pre = 'base_' if c == 'base' else f'{c}|'
        T = dict(pt=Z[pre + 'pt'].astype(np.float64), eta=Z[pre + 'eta'].astype(np.float64), phi=Z[pre + 'phi'].astype(np.float64),
                 m=np.zeros(Z['base_pt'].shape), ax_eta=np.zeros(J), ax_phi=np.zeros(J), ext={})
        X = H.hls_prep(H.sortT(T))
        for k, n in nets.items(): out[f'lg|{k}|{c}'] = np.asarray(n.run(X)['logits'], np.float32)
        log('jedi', c, ' '.join(f'{k} acc {(out[f"lg|{k}|{c}"].argmax(1) == y).mean():.4f}' for k in nets))
    np.savez_compressed(RES / f'h80_hls_jedi{tag}.npz', y=y, **out); log('wrote h80_hls_jedi')


# ================================================================================================================ stats
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def signflip(d, P=NPERM, seed=0):
    d = np.asarray(d, float); r = np.random.default_rng(seed); obs_ = abs(d.mean()); cnt = 0
    for c in range(0, P, 1000):
        sg = r.choice(np.array([-1.0, 1.0]), (min(1000, P - c), len(d))); cnt += int((np.abs(sg @ d / len(d)) >= obs_ - 1e-15).sum())
    return (cnt + 1) / (P + 1)


def mcnemar(a, b):
    from scipy.stats import binomtest
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    return dict(only_NS=n01, only_FREE=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0)


def load_ds(ds, tag):
    Z = dict(np.load(RES / f'h80_{ds}{tag}.npz')); R = json.loads((RES / f'h80_{ds}{tag}.json').read_text())
    if ds == 'hls' and (RES / f'h80_hls_jedi{tag}.npz').exists():
        from interp.h71_energy_scale_panel import logodds_all
        Jd = dict(np.load(RES / f'h80_hls_jedi{tag}.npz')); y = Z['y'].astype(int); r = np.arange(len(y))
        assert np.array_equal(Jd['y'].astype(int), y)
        for k, v in Jd.items():
            if not k.startswith('lg|'): continue
            _, mn, c = k.split('|'); Z[f'lo|{mn}|{c}'] = logodds_all(v)[r, y].astype(np.float32)
            if c == 'base':
                lg = v.astype(float); cc = lg.argmax(1); a, ab = sig_auc(lg, y, [2, 3, 4], [0, 1])
                R['models'][mn] = dict(role='perm' if mn.endswith('_perm') else 'trained', arch='JEDI', acc=float((cc == y).mean()),
                                       known='0.8066 (H64) / 0.8100 (H12)' if mn == 'JEDI' else None, auc_sig=a, auc_by_class=ab,
                                       per_class={R['classes'][k]: float((cc[y == k] == k).mean()) for k in range(5)}, logit_std=float(lg.std(0).mean()))
    return Z, R


PERM_OF = dict(top=dict(ParT_kin='ParT_kin_perm', PN_kin='PN_kin_perm', PN_kin_s1='PN_kin_perm', PFN='PFN_perm', PFN_s1='PFN_perm', EFN='EFN_perm',
                        LorentzNet='ParT_kin_perm'),
               jc=dict(ParT_full='ParT_full_perm', ParT_kin='ParT_kin_perm', ParT_kin_s0='ParT_kin_perm', ParT_kin_s1='ParT_kin_perm',
                       ParT_kin_s2='ParT_kin_perm', PN_full='PN_full_perm', PN_kin='PN_kin_perm'),
               qg=dict(part_kin='part_kin_perm', part_kin_s1='part_kin_perm', pnet_kin='pnet_kin_perm', lorentznet='part_kin_perm',
                       pfn='pfn_perm', efn='pfn_perm'),
               hls=dict(JEDI='JEDI_perm'))


def part_stats(a):
    tag = '_smoke' if a.smoke else ''; S = dict(B=NB, P=NPERM, ds={}); tests = []
    for ds in a.ds.split(','):
        if not (RES / f'h80_{ds}{tag}.npz').exists(): log('missing', ds); continue
        Z, R = load_ds(ds, tag); y = Z['y'].astype(int); J = len(y); rng = np.random.default_rng(SEEDS[ds] + 100)
        idx = rng.integers(0, J, (NB, J))
        models = sorted({k.split('|')[1] for k in Z if k.startswith('lo|') and k.endswith('|base')})
        c2 = R['c2st']
        prim = None
        for e in EPS:   # largest step at which every arm has C2ST <= 0.65
            if all(c2[f'{a_}@{e}']['auc'] <= 0.65 for a_ in ARMS): prim = e; break
        # per arm: largest step at which that arm and FREE both have C2ST <= 0.65; otherwise the smallest step (flagged)
        prim_arm = {}
        for a_ in ARMS:
            if a_ == 'FREE': continue
            ok_e = [e for e in EPS if c2[f'{a_}@{e}']['auc'] <= 0.65 and c2[f'FREE@{e}']['auc'] <= 0.65]
            prim_arm[a_] = dict(eps=ok_e[0] if ok_e else EPS[-1], on_manifold=bool(ok_e))
        G = S['ds'][ds] = dict(n=J, models_info=R['models'], c2st=c2, c2st_blocks=R.get('c2st_blocks'), edits=R['edits'], primary_eps=prim,
                               primary_eps_by_arm=prim_arm, kappa={}, sample=R['sample'])
        sig = [1] if ds in ('top', 'qg') else None
        for mn in models:
            s0 = Z[f'lo|{mn}|base'].astype(float)
            # frozen 50 % working point: binary -> signal score at the median of the signal jets; multi-class -> own-class
            # log-odds at the median of that class's jets
            if sig:
                sgn = np.where(y == 1, 1.0, -1.0); thr = np.median((s0 * sgn)[y == 1])
                call = lambda s: (s * sgn) > thr
            else:
                thr = np.array([np.median(s0[y == k]) if (y == k).any() else 0 for k in range(y.max() + 1)])
                call = lambda s: s > thr[y]
            c0 = call(s0); role = R['models'].get(mn, {}).get('role', 'ref')
            if f'lo|{mn}|FLOOR@{FLOOR_EPS}' in Z:
                fl = np.abs(Z[f'lo|{mn}|FLOOR@{FLOOR_EPS}'].astype(float) - s0)
                G.setdefault('floor', {})[mn] = dict(abs_ds=float(fl.mean()), ci=np.percentile(_boot(fl, idx), [2.5, 97.5]).tolist(),
                                                     call_change=float((call(Z[f'lo|{mn}|FLOOR@{FLOOR_EPS}'].astype(float)) != c0).mean()))
            for e in EPS:
                f = np.abs(Z[f'lo|{mn}|FREE@{e}'].astype(float) - s0)
                for arm, M in ARMS.items():
                    if arm == 'FREE': continue
                    ok = Z[f'ok|{arm}@{e}'] & Z[f'ok|FREE@{e}']
                    if ok.sum() < 20: continue
                    d = np.abs(Z[f'lo|{mn}|{arm}@{e}'].astype(float) - s0)
                    num = _boot(np.where(ok, d, 0), idx); den = _boot(np.where(ok, f, 0), idx)
                    kap = d[ok].mean() / max(f[ok].mean(), 1e-12)
                    if f'lo|{mn}|FLOOR@{FLOOR_EPS}' in Z:   # floor-corrected ratio: numerical floor subtracted from both arms
                        flr = np.abs(Z[f'lo|{mn}|FLOOR@{FLOOR_EPS}'].astype(float) - s0); bf = _boot(np.where(ok, flr, 0), idx)
                        okb = np.maximum(_boot(ok.astype(float), idx), 1e-9); fm = flr[ok].mean()
                        kc = (d[ok].mean() - fm) / max(f[ok].mean() - fm, 1e-12)
                        kcb = (num - bf) / np.maximum(den - bf, 1e-12)
                        floor_ = dict(kappa_floorcorr=float(kc), kappa_floorcorr_ci=np.percentile(kcb, [2.5, 97.5]).tolist(),
                                      floor_share_FREE=float(fm / max(f[ok].mean(), 1e-12)))
                    else: floor_ = {}
                    chA = call(Z[f'lo|{mn}|{arm}@{e}'].astype(float)) != c0; chF = call(Z[f'lo|{mn}|FREE@{e}'].astype(float)) != c0
                    r_ = dict(n_ok=int(ok.sum()), kappa=float(kap), kappa_ci=np.percentile(num / np.maximum(den, 1e-12), [2.5, 97.5]).tolist(),
                              abs_ds_NS=float(d[ok].mean()), abs_ds_NS_ci=np.percentile(num / np.maximum(_boot(ok.astype(float), idx), 1e-9), [2.5, 97.5]).tolist(),
                              abs_ds_FREE=float(f[ok].mean()), abs_ds_FREE_ci=np.percentile(den / np.maximum(_boot(ok.astype(float), idx), 1e-9), [2.5, 97.5]).tolist(),
                              mean_ds_NS=float((Z[f'lo|{mn}|{arm}@{e}'].astype(float) - s0)[ok].mean()),
                              call_change_NS=float(chA[ok].mean()), call_change_FREE=float(chF[ok].mean()), mcnemar=mcnemar(chA[ok], chF[ok]), role=role, **floor_)
                    if role == 'trained' or mn in ('GBDT_B6', 'LR_pTD') or mn.startswith('EFN') or mn == 'efn':
                        r_['p_kappa_eq_1'] = signflip((d - f)[ok], seed=1)
                    pm = PERM_OF.get(ds, {}).get(mn)
                    if role == 'trained' and pm and f'lo|{pm}|base' in Z:
                        p0 = Z[f'lo|{pm}|base'].astype(float); d0 = np.abs(Z[f'lo|{pm}|{arm}@{e}'].astype(float) - p0); f0 = np.abs(Z[f'lo|{pm}|FREE@{e}'].astype(float) - p0)
                        if f0[ok].mean() > 1e-6:
                            u = d[ok] / f[ok].mean() - d0[ok] / f0[ok].mean()
                            r_.update(null=pm, kappa_null=float(d0[ok].mean() / f0[ok].mean()), p_vs_null=signflip(u, seed=2))
                        else: r_.update(null=pm, kappa_null=None, null_note='permuted net constant under FREE')
                    G['kappa'][f'{mn}|{arm}@{e}'] = r_
                    if role == 'trained' and e == prim_arm[arm]['eps']:
                        tests.append((ds, mn, arm, e, 'kappa=1', r_['p_kappa_eq_1']))
                        if r_.get('p_vs_null') is not None: tests.append((ds, mn, arm, e, 'kappa=null', r_['p_vs_null']))
        # H3 check: kappa of trained nets vs C2ST across arms x eps
        xs, ys_ = [], []
        for k, v in G['kappa'].items():
            mn, ce = k.split('|')
            if v['role'] == 'trained': xs.append(c2[ce]['auc']); ys_.append(v['kappa'])
        G['corr_kappa_c2st'] = float(np.corrcoef(xs, ys_)[0, 1]) if len(xs) > 2 else None
        log(ds, 'primary eps', prim, ' '.join(f'{k} {v["kappa"]:.3f}' for k, v in G['kappa'].items() if k.endswith(f'NS6@{prim or EPS[-1]}')))
    if tests:
        adj = holm([t[-1] for t in tests])
        S['holm'] = [dict(ds=t[0], model=t[1], arm=t[2], eps=t[3], test=t[4], p=t[5], p_holm=float(q)) for t, q in zip(tests, adj)]
        S['n_tests'] = len(tests); S['n_holm_005'] = int((adj < 0.05).sum())
    (RES / f'h80_stats{tag}.json').write_text(json.dumps(S, indent=1, default=float)); log('wrote h80_stats', S.get('n_holm_005'), '/', S.get('n_tests'))


def _boot(x, idx):
    return np.array([x[i].mean() for i in idx])


# ================================================================================================================== fig
def part_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tag = '_smoke' if a.smoke else ''; S = json.loads((RES / f'h80_stats{tag}.json').read_text()); X = dict(np.load(RES / f'h80_example_top{tag}.npz'))
    fig = plt.figure(figsize=(16, 7.2)); gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 0.02])
    ax = fig.add_subplot(gs[:, 0])
    def rel(L):
        pt, eta, phi = L[:, 0], L[:, 1], L[:, 2]; px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        ej = np.arcsinh(pz.sum() / np.hypot(px.sum(), py.sum())); pj = np.arctan2(py.sum(), px.sum())
        return eta - ej, (phi - pj + np.pi) % (2 * np.pi) - np.pi
    x0, y0 = rel(X['base']); x1, y1 = rel(X['edit']); w0 = X['base'][:, 0] / X['base'][:, 0].sum(); w1 = X['edit'][:, 0] / X['edit'][:, 0].sum()
    ax.scatter(x0, y0, s=6 + 900 * w0, facecolors='none', edgecolors='0.45', lw=0.9, label='real top jet')
    ax.scatter(x1, y1, s=6 + 900 * w1, color='C3', alpha=0.55, lw=0, label='after NS$_6$ edit (ε = 0.01)')
    for i in range(len(x0)): ax.plot([x0[i], x1[i]], [y0[i], y1[i]], color='C3', lw=0.7)
    t0, t1 = X['tau0'], X['tau1']
    ax.set_title(f'Top Landscape top jet: every particle moved (RMS 0.01 in ΔR, 3 % in pT)\n'
                 f'held: m = {float(X["m0"]):.1f} GeV, pT = {float(X["jpt0"]):.0f} GeV, axis, N = {len(x0)}, 15 τ$_N^β$ (N ≤ 5)\n'
                 f'τ32 (β=1): {t0[2, 1] / t0[1, 1]:.4f} → {t1[2, 1] / t1[1, 1]:.4f};  τ21: {t0[1, 1] / t0[0, 1]:.4f} → {t1[1, 1] / t1[0, 1]:.4f}', fontsize=9)
    ax.set_xlabel('Δη from jet axis'); ax.set_ylabel('Δφ from jet axis'); ax.set_aspect('equal')
    lim = max(0.85, np.abs(np.r_[x0, y0]).max() * 1.05); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8, loc='lower right', title='marker area ∝ pT', title_fontsize=8)
    j0 = int(np.argmax(w0)); cx, cy = x0[j0], y0[j0]; h_ = 0.04
    ins = ax.inset_axes([0.62, 0.62, 0.36, 0.36])
    ins.scatter(x0, y0, s=20 + 3000 * w0, facecolors='none', edgecolors='0.45', lw=0.9)
    ins.scatter(x1, y1, s=20 + 3000 * w1, color='C3', alpha=0.55, lw=0)
    for i in range(len(x0)): ins.plot([x0[i], x1[i]], [y0[i], y1[i]], color='C3', lw=0.8)
    ins.set_xlim(cx - h_, cx + h_); ins.set_ylim(cy - h_, cy + h_); ins.set_aspect('equal'); ins.tick_params(labelsize=6)
    ins.set_title(f'zoom ±{h_} around the hardest particle', fontsize=7); ax.indicate_inset_zoom(ins, edgecolor='0.5')
    names = dict(top='Top Landscape (top vs QCD)', jc='JetClass (t, W, Z, QCD)', qg='Quark–gluon (Pythia)', hls='hls4ml (5 classes)')
    for i, ds in enumerate(('top', 'jc', 'qg', 'hls')):
        axd = fig.add_subplot(gs[i // 2, 1 + i % 2]); G = S['ds'].get(ds)
        if not G: axd.set_visible(False); continue
        e = G['primary_eps_by_arm']['NS6']['eps']; Ms = [3, 4, 6]
        for mn, inf in G['models_info'].items():
            role = inf.get('role')
            if role not in ('trained', 'perm') and mn not in ('GBDT_B6', 'LR_pTD'): continue
            ks = [G['kappa'].get(f'{mn}|NS{M}@{e}') for M in Ms]
            if any(k is None for k in ks): continue
            v = [k.get('kappa_floorcorr', k['kappa']) for k in ks]
            if role == 'trained':
                st = dict(color='C0' if not mn.lower().startswith('efn') else 'C2', lw=1.2, alpha=0.8, marker='o', ms=3)
            elif role == 'perm': st = dict(color='0.6', lw=0.8, ls='--', marker='x', ms=3)
            elif mn == 'GBDT_B6': st = dict(color='k', lw=1.5, marker='s', ms=4)
            else: st = dict(color='C1', lw=1.5, marker='^', ms=4)
            axd.plot(Ms, v, **st)
        axd.axhline(1, color='0.8', lw=0.8); axd.set_ylim(-0.03, 1.65); axd.set_xticks(Ms)
        sat = {int(k[6:]): v['auc_sig'] for k, v in G['models_info'].items() if k.startswith('GBDT_B')}
        tr = [v['auc_sig'] for k, v in G['models_info'].items() if v.get('role') == 'trained' and not k.endswith(('_shuf',))]
        ax2 = axd.twinx(); xs = sorted(sat); ax2.plot(xs, [sat[x] for x in xs], color='C4', lw=1, ls=':', marker='.')
        if tr: ax2.axhspan(min(tr), max(tr), color='C4', alpha=0.08)
        ax2.set_ylabel('AUC (signal vs QCD)', fontsize=7, color='C4'); ax2.tick_params(labelsize=7, colors='C4')
        axd.set_title(f'{names[ds]}, ε = {e}, C2ST(NS6) {G["c2st"][f"NS6@{e}"]["auc"]:.2f}', fontsize=8.5)
        axd.set_xlabel('M (basis B$_M$ held: τ$_N$, N < M)', fontsize=8); axd.set_ylabel('κ$_M$ (floor-subtracted)\n|Δs| basis held / |Δs| FREE', fontsize=8)
        axd.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color='C0', marker='o', label='trained networks'), Line2D([], [], color='C2', marker='o', label='EFN (IRC-safe)'),
         Line2D([], [], color='0.6', ls='--', marker='x', label='weight-permuted networks'), Line2D([], [], color='k', marker='s', label='GBDT on B$_6$ (solver check)'),
         Line2D([], [], color='C1', marker='^', label='LR(pT$^D$) (IRC-unsafe)'), Line2D([], [], color='C4', ls=':', marker='.', label='right axis: GBDT(B$_M$) AUC; band = networks')]
    fig.legend(handles=h, loc='lower center', ncol=6, fontsize=8, frameon=False)
    fig.suptitle('Holding the N-subjettiness basis B$_M$ (plus m, pT, axis, N) fixed while moving every particle: response relative to an edit that moves the basis', fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95)); p = RES / f'h80_nsub_basis{tag}.png'; fig.savefig(p, dpi=130); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True, choices=['run', 'jedi', 'stats', 'fig'])
    ap.add_argument('--ds', default='top,jc,qg,hls'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); RES.mkdir(parents=True, exist_ok=True)
    dict(run=part_run, jedi=part_jedi, stats=part_stats, fig=part_fig)[a.part](a)
