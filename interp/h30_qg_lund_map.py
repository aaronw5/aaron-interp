"""H30 (loop D; claim D7): Lund-plane response map of quark-gluon taggers. Where in (Delta R, kt) does moving one existing soft
particle change the gluon logit, and does the map follow the gluon/quark primary Lund-density ratio rho_g / rho_q, girth
(sum z_i R_i) or pT^D (sqrt(sum z_i^2))?

Samples: QG test jets, Pythia 8.226 (interp.data_qg test split) and Herwig 7.1 (Zenodo 3066475 files 18-19), 2000 quark +
  2000 gluon jets per generator, random rows, seed 3001 (interp.h14_qg_generator.load_sample).
Models (logit l = log p(gluon) - log p(quark)): as H24. Pythia-trained part_kin, pnet_kin, pfn_k, pfn_t, efn_k, efn_t
  (EFN IRC-safe), part_kin_perm (permuted-weight null); Herwig-trained hw_part_kin, hw_pnet_kin.
Edit (one particle per jet). Selected particle i: random among particles with z_i = pT_i / sum pT in [0.005, 0.02] (not the
  hardest). Source coordinates about the E-scheme axis: Delta R0, kt0 = pT_i Delta R0, azimuth psi0.
  main m<r><k>: grid cell (Delta R_r, kt_k), Delta R centres of 6 log bins in [0.02, 0.4], kt centres of 6 log bins in
    [0.3, 10] GeV. pT_i -> kt_k / Delta R_r; the hardest particle gives / takes the difference (sum pT and N fixed); the
    particle is placed at distance Delta R_r from the edited jet's own E-scheme axis along psi0 (axis recomputed 6 times).
    Cell skipped for the jet if the new z > 0.10 or the hardest particle would lose > 50 % of its pT.
  c1 c<r><k>: the same particle, pT unchanged, rotated about the axis at its own Delta R0 (radial coordinates unchanged) by
    the angle whose chord equals the main edit's displacement (pi if the displacement exceeds 2 Delta R0).
  c2 r<r><k>: the same particle, pT unchanged, displaced by the main edit's |Delta position| in a random direction
    (resampled until it stays within Delta R < 0.4 of the axis).
Steps
  --step eval  (GPU) logits + observables per condition -> physics/h30_logits_<gen>.npz, h30_edits_<gen>.npz, h30_models.json
  --step ref   (CPU, venv-ef) --part lund: C/A primary Lund densities per class and generator (numpy declustering of 20k
               train jets per class), LR(Lund) (Poisson LR sum_e ln rho_g/rho_q over primary emissions), LR(girth), LR(pT^D),
               LR(N) on every condition -> physics/h30_ref_<gen>.npz, h30_lund.npz
               --part efp: GBDT(EFP d<=5 + N) trained on 15k train jets per class, on 500 + 500 jets of the sample for base,
               main and c1 -> physics/h30_efp_<gen>.npz
  --step stats (CPU) -> physics/h30_run.json
  --step fig   (CPU) -> physics/h30_lund_map.png, h30_lund_fit.png
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
SEED = 3001
ZLO, ZHI, ZCAP, HMIN = 0.005, 0.02, 0.10, 0.5
RE = np.geomspace(0.02, 0.4, 7); RC = np.sqrt(RE[:-1] * RE[1:])
KE = np.geomspace(0.3, 10.0, 7); KC = np.sqrt(KE[:-1] * KE[1:])
CELLS = [(r, k) for r in range(6) for k in range(6)]
# fine table for the Lund densities: ln(1/dR) and ln(kt / GeV)
FR = np.linspace(np.log(1 / 0.8), np.log(1 / 0.002), 16); FK = np.linspace(np.log(0.02), np.log(100.0), 18)
TRAINED = dict(part_kin='py', pnet_kin='py', pfn_k='py', efn_k='py', pfn_t='py', efn_t='py', part_kin_perm='py',
               hw_part_kin='hw', hw_pnet_kin='hw')
PAIRS = (('hw_part_kin', 'part_kin'), ('hw_pnet_kin', 'pnet_kin'))
wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi


def _rng(gen, name):
    return np.random.default_rng([SEED, zlib.crc32(gen.encode()), zlib.crc32(name.encode())])


def cn(kind, r, k):
    return f'{kind}{r}{k}'


# ----------------------------------------------------------------------------------------------------------------- edits
def axis(a):
    pt, eta, phi = a[:, 0], a[:, 1], a[:, 2]
    jx, jy, jz = (pt * np.cos(phi)).sum(), (pt * np.sin(phi)).sum(), (pt * np.sinh(eta)).sum()
    return np.arcsinh(jz / np.hypot(jx, jy)), np.arctan2(jy, jx)


def _place(b, i, R, psi, it=6):
    for _ in range(it):
        je, jp = axis(b); b[i, 1] = je + R * np.cos(psi); b[i, 2] = wrap(jp + R * np.sin(psi))
    return b


def _dr_axis(b, i):
    je, jp = axis(b); return float(np.hypot(b[i, 1] - je, wrap(b[i, 2] - jp)))


def make_edits(L0, gen):
    """-> meta dict (sel, dR0, kt0, z0, pTJ) and P: cond -> (J, 5) [pT_i, eta_i, phi_i, pT_hardest, ok]"""
    J = len(L0); rs = _rng(gen, 'select')
    sel = np.full(J, -1); dR0 = np.zeros(J); kt0 = np.zeros(J); z0 = np.zeros(J); pTJ = np.zeros(J); psi0 = np.zeros(J)
    for j, a in enumerate(L0):
        P = a[:, 0].sum(); z = a[:, 0] / P; c = np.flatnonzero((z >= ZLO) & (z <= ZHI)); c = c[c > 0]
        pTJ[j] = P
        if len(c) == 0: continue
        i = int(rs.choice(c)); sel[j] = i; je, jp = axis(a)
        de, dp = a[i, 1] - je, wrap(a[i, 2] - jp); dR0[j] = np.hypot(de, dp); psi0[j] = np.arctan2(dp, de)
        kt0[j] = a[i, 0] * dR0[j]; z0[j] = z[i]
    P = {}
    for (r, k) in CELLS:
        M = np.zeros((J, 5)); C1 = np.zeros((J, 5)); C2 = np.zeros((J, 5))
        r1 = _rng(gen, cn('c', r, k)); r2 = _rng(gen, cn('r', r, k))
        for j, a in enumerate(L0):
            i = sel[j]
            sgn = r1.choice([-1.0, 1.0]); ths = r2.uniform(0, 2 * np.pi, 40)       # drawn for every jet (stream alignment)
            if i < 0: continue
            pnew = KC[k] / RC[r]; d = pnew - a[i, 0]; ph = a[0, 0] - d
            if pnew / pTJ[j] > ZCAP or ph < HMIN * a[0, 0]: continue
            b = a.copy(); b[i, 0] = pnew; b[0, 0] = ph; b = _place(b, i, RC[r], psi0[j])
            M[j] = [b[i, 0], b[i, 1], b[i, 2], b[0, 0], 1]
            disp = float(np.hypot(b[i, 1] - a[i, 1], wrap(b[i, 2] - a[i, 2])))
            al = np.pi if disp >= 2 * dR0[j] else 2 * np.arcsin(disp / (2 * dR0[j]))
            c = _place(a.copy(), i, dR0[j], psi0[j] + sgn * al)
            C1[j] = [c[i, 0], c[i, 1], c[i, 2], c[0, 0], 1]
            for th in ths:
                c = a.copy(); c[i, 1] = a[i, 1] + disp * np.cos(th); c[i, 2] = wrap(a[i, 2] + disp * np.sin(th))
                if _dr_axis(c, i) < 0.4: C2[j] = [c[i, 0], c[i, 1], c[i, 2], c[0, 0], 1]; break
        P[cn('m', r, k)] = M; P[cn('c', r, k)] = C1; P[cn('r', r, k)] = C2
    meta = dict(sel=sel, dR0=dR0, kt0=kt0, z0=z0, pTJ=pTJ, psi0=psi0)
    return meta, P


def apply(L0, sel, prm):
    """edited lists (pT-sorted); jets with ok = 0 are returned unedited"""
    out = []
    for j, a in enumerate(L0):
        if prm[j, 4] < 0.5: out.append(a); continue
        b = a.copy(); i = sel[j]; b[i, :3] = prm[j, :3]; b[0, 0] = prm[j, 3]
        out.append(b[np.argsort(-b[:, 0], kind='stable')])
    return out


def conds(P):
    return ['base'] + [c for c in P if P[c][:, 4].any()]


def load(gen, per_class):
    from interp import h14_qg_generator as H14
    return H14.load_sample(gen, per_class, SEED)


def moved_coords(L0, sel, prm):
    """Delta R of the moved particle to the edited jet's axis and kt = pT_i Delta R"""
    R = np.zeros(len(L0)); K = np.zeros(len(L0))
    for j, a in enumerate(L0):
        if prm[j, 4] < 0.5: continue
        b = a.copy(); i = sel[j]; b[i, :3] = prm[j, :3]; b[0, 0] = prm[j, 3]; R[j] = _dr_axis(b, i); K[j] = b[i, 0] * R[j]
    return R, K


# ------------------------------------------------------------------------------------------------------------------- eval
def step_eval(a):
    import torch
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    from interp.h24_qg_generator_models import basic_obs
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    M = H.Models(dev)
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')): M.add_weaver(name, arch, False)
    M.add_permuted('part_kin_perm', 'part_kin')
    for kind in ('efn', 'pfn'):
        H14.add_keras_ef(M, kind)
        net = H.make_ef(kind); net.load_state_dict(torch.load(V / 'models' / 'qg' / f'h5_{kind}_torch.pt', map_location='cpu'))
        M.add_torch(f'{kind}_t', net, json.loads((V / 'models' / 'qg' / f'h5_{kind}_torch.json').read_text()))
    for name, arch in (('part_kin', 'part'), ('pnet_kin', 'pn')):
        d = V / 'models' / 'qg_herwig' / name
        if (d / 'net_best_epoch_state.pt').exists(): H14.add_weaver_dir(M, 'hw_' + name, arch, False, d)
        else: log(f'hw_{name}: no checkpoint')
    names = list(M.m); log('models', names); tag = '_smoke' if a.per_class < 2000 else ''
    for gen in a.gens.split(','):
        L0, y, info = load(gen, a.per_class)
        log(f'{gen}: {len(y)} jets, quark frac {y.mean():.3f}; device {dev}')
        meta, P = make_edits(L0, gen); s = meta['sel'] >= 0
        log(gen, f'edits built: selected particle in {s.sum()} jets; z0 mean {meta["z0"][s].mean():.4f}, dR0 median '
                 f'{np.median(meta["dR0"][s]):.3f}, kt0 median {np.median(meta["kt0"][s]):.3f} GeV; valid main cells per jet '
                 f'{np.mean([P[cn("m", r, k)][:, 4].sum() for r, k in CELLS]):.0f} avg')
        np.savez_compressed(RES / f'h30_edits_{gen}{tag}.npz', y=y, idx=info['idx'], **meta, **{f'prm|{c}': v for c, v in P.items()})
        out = dict(y=y, idx=info['idx'], **meta)
        for c in conds(P):
            Le = L0 if c == 'base' else apply(L0, meta['sel'], P[c]); Jt = H.from_list(Le, Lpad=180)
            for nm in names: out[f'{c}|{nm}'] = M.logit(nm, Jt).astype(np.float32)
            O = basic_obs(Le); out.update({f'obs|{c}|{k}': v.astype(np.float32) for k, v in O.items()})
            if c == 'base':
                log(gen, 'unedited', '; '.join(f'{nm} {H.roc_metrics(out[f"base|{nm}"].astype(float), y)}' for nm in names))
                continue
            ok = P[c][:, 4] > 0.5; out[f'ok|{c}'] = ok
            R, K = moved_coords(L0, meta['sel'], P[c]); out[f'R|{c}'] = R.astype(np.float32); out[f'kt|{c}'] = K.astype(np.float32)
            if c[0] == 'm' or a.per_class < 2000:
                log(gen, c, f'n {ok.sum()}, dR {R[ok].mean():.4f} kt {K[ok].mean():.3f} dN {np.mean((O["N"] - out["obs|base|N"])[ok]):+.2f} '
                    f'dgirth {np.mean((O["girth"] - out["obs|base|girth"])[ok]):+.5f} dptd {np.mean((O["ptd"] - out["obs|base|ptd"])[ok]):+.5f} | '
                    + ' '.join(f'{nm} {np.mean((out[f"{c}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in names))
        np.savez_compressed(RES / f'h30_logits_{gen}{tag}.npz', **out); log('wrote', RES / f'h30_logits_{gen}{tag}.npz')
    (RES / f'h30_models{tag}.json').write_text(json.dumps(M.info, indent=1, default=str))


# ------------------------------------------------------------------------------------------------------------ C/A Lund
def ca_primary(a):
    """Cambridge/Aachen clustering of all particles (pT, y, phi) into one jet (numpy), then primary declustering along the
    harder branch -> array (n_e, 3): ln(1/dR), ln(kt / GeV), z with kt = pT_soft dR"""
    n = len(a)
    if n < 2: return np.zeros((0, 3))
    pt, yy, ph = a[:, 0].astype(float), a[:, 1].astype(float), a[:, 2].astype(float)
    P4 = np.stack([pt * np.cos(ph), pt * np.sin(ph), pt * np.sinh(yy), pt * np.cosh(yy)], 1)
    mom = np.zeros((2 * n, 4)); mom[:n] = P4; ch = np.full((2 * n, 2), -1)
    Y, PH, ids = yy.copy(), ph.copy(), np.arange(n); act = np.ones(n, bool)
    D = (Y[:, None] - Y[None]) ** 2 + wrap(PH[:, None] - PH[None]) ** 2; np.fill_diagonal(D, np.inf); nid = n
    for _ in range(n - 1):
        f = int(np.argmin(D)); i, j = divmod(f, n)
        v = mom[ids[i]] + mom[ids[j]]; mom[nid] = v; ch[nid] = (ids[i], ids[j]); ids[i] = nid; nid += 1
        act[j] = False; D[j, :] = np.inf; D[:, j] = np.inf
        ptv = np.hypot(v[0], v[1]); Y[i] = 0.5 * np.log((v[3] + v[2]) / max(v[3] - v[2], 1e-12)) if ptv > 0 else 0; PH[i] = np.arctan2(v[1], v[0])
        dd = (Y[i] - Y) ** 2 + wrap(PH[i] - PH) ** 2; dd[~act] = np.inf; dd[i] = np.inf; D[i, :] = dd; D[:, i] = dd
    em = []; node = nid - 1
    rap = lambda v: 0.5 * np.log((v[3] + v[2]) / max(v[3] - v[2], 1e-12))
    while ch[node, 0] >= 0:
        u, w = mom[ch[node, 0]], mom[ch[node, 1]]; pu, pw = np.hypot(u[0], u[1]), np.hypot(w[0], w[1])
        if pw > pu: u, w, pu, pw = w, u, pw, pu; node = ch[node, 1]
        else: node = ch[node, 0]
        dR = np.hypot(rap(u) - rap(w), wrap(np.arctan2(u[1], u[0]) - np.arctan2(w[1], w[0])))
        if dR > 0: em.append((np.log(1 / dR), np.log(max(pw * dR, 1e-6)), pw / (pu + pw)))
    return np.array(em).reshape(-1, 3)


def _lund_chunk(L):
    return [ca_primary(a[:, :3]) for a in L]


def lund_all(L, pool):
    ch = [L[s:s + 100] for s in range(0, len(L), 100)]
    return [e for part in pool.map(_lund_chunk, ch) for e in part]


def fbin(lr, lk):
    return (np.clip(np.searchsorted(FR, lr, side='right') - 1, 0, len(FR) - 2),
            np.clip(np.searchsorted(FK, lk, side='right') - 1, 0, len(FK) - 2))


def lund_table(E, y):
    H = {}
    for c in (0, 1):
        h = np.zeros((len(FR) - 1, len(FK) - 1)); cg = np.zeros((6, 6)); nj = int((y == c).sum())
        for e, yy in zip(E, y):
            if yy != c or len(e) == 0: continue
            bi, bk = fbin(e[:, 0], e[:, 1]); np.add.at(h, (bi, bk), 1)
            dr, kt = np.exp(-e[:, 0]), np.exp(e[:, 1])
            ir = np.searchsorted(RE, dr, side='right') - 1; ik = np.searchsorted(KE, kt, side='right') - 1
            m = (ir >= 0) & (ir < 6) & (ik >= 0) & (ik < 6); np.add.at(cg, (ir[m], ik[m]), 1)
        H[c] = (h, cg, nj)
    T = np.log((H[0][0] + 0.5) / H[0][2]) - np.log((H[1][0] + 0.5) / H[1][2])
    area = np.log(RE[1:] / RE[:-1])[:, None] * np.log(KE[1:] / KE[:-1])[None]
    rho = {c: H[c][1] / H[c][2] / area for c in (0, 1)}
    return T, rho, dict(n_g=H[0][2], n_q=H[1][2])


def lund_lr(E, T):
    out = np.zeros(len(E))
    for j, e in enumerate(E):
        if len(e): bi, bk = fbin(e[:, 0], e[:, 1]); out[j] = T[bi, bk].sum()
    return out


def step_ref(a):
    from multiprocessing import Pool
    from interp import h14_qg_generator as H14
    from interp.h24_qg_generator_models import basic_obs, _lr_hist
    RES.mkdir(parents=True, exist_ok=True); smoke = a.per_class < 2000; tag = '_smoke' if smoke else ''
    pool = Pool(a.cpu)
    if a.part == 'lund':
        LT, lrinfo = H14.lrn_tables(); Tab, rho, hist, info = {}, {}, {}, {}
        for gen in ('py', 'hw'):
            Lt, yt, Nt = H14.efp_train_lists(gen, n_per_class=1000 if smoke else 20000, seed=SEED + 1)
            Et = lund_all(Lt, pool); Tab[gen], rho[gen], info[gen] = lund_table(Et, yt)
            Ot = basic_obs([np.c_[x, np.zeros(len(x))] for x in Lt])
            bp = np.quantile(Ot['ptd'], np.linspace(0, 1, 41)); bp[0], bp[-1] = 0, 1.01
            bg = np.quantile(Ot['girth'], np.linspace(0, 1, 41)); bg[0], bg[-1] = 0, 10
            hist[gen] = dict(ptd=(bp, _lr_hist(Ot['ptd'], yt, bp)), girth=(bg, _lr_hist(Ot['girth'], yt, bg)))
            ne = np.array([len(e) for e in Et])
            info[gen].update(mean_primary_q=float(ne[yt == 1].mean()), mean_primary_g=float(ne[yt == 0].mean()))
            log(f'{gen}: Lund densities from {len(yt)} train jets; primary emissions q {ne[yt == 1].mean():.2f} g {ne[yt == 0].mean():.2f}; '
                f'grid ln(rho_g/rho_q):\n' + np.array2string(np.log(rho[gen][0] / np.maximum(rho[gen][1], 1e-9)), precision=2))
        np.savez(RES / f'h30_lund{tag}.npz', FR=FR, FK=FK, RE=RE, KE=KE, **{f'T_{g}': Tab[g] for g in Tab},
                 **{f'rho_{c}_{g}': rho[g][c] for g in rho for c in (0, 1)})
        look = lambda gen, k, x: hist[gen][k][1][np.clip(np.searchsorted(hist[gen][k][0], x, side='right') - 1, 0, 39)]
        for gen in ('py', 'hw'):
            L0, y, inf = load(gen, a.per_class); meta, P = make_edits(L0, gen)
            fe = RES / f'h30_edits_{gen}{tag}.npz'
            if fe.exists():
                Z = np.load(fe); assert (Z['sel'] == meta['sel']).all()
                for c in P: assert np.allclose(Z[f'prm|{c}'], P[c]), c
                log(gen, 'edits identical to the eval step')
            out = dict(y=y)
            for c in conds(P):
                Le = L0 if c == 'base' else apply(L0, meta['sel'], P[c]); E = lund_all(Le, pool); O = basic_obs(Le)
                for tg in ('py', 'hw'):
                    out[f'{c}|lund_{tg}'] = lund_lr(E, Tab[tg]).astype(np.float32)
                    out[f'{c}|lrg_{tg}'] = look(tg, 'girth', O['girth']).astype(np.float32)
                    out[f'{c}|lrp_{tg}'] = look(tg, 'ptd', O['ptd']).astype(np.float32)
                    out[f'{c}|lrn_{tg}'] = LT[tg][np.minimum(O['N'].astype(int), 299)].astype(np.float32)
                if c[0] == 'm':
                    ok = P[c][:, 4] > 0.5
                    log(gen, c, ' '.join(f'{k.split("|")[1]} {np.mean((out[k] - out["base|" + k.split("|")[1]])[ok]):+.4f}'
                                         for k in out if k.startswith(c + '|')))
            np.savez_compressed(RES / f'h30_ref_{gen}{tag}.npz', **out); log('wrote', RES / f'h30_ref_{gen}{tag}.npz')
        (RES / f'h30_lund{tag}.json').write_text(json.dumps(dict(lrn=lrinfo, lund=info), indent=1))
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        feat = lambda F, N: np.c_[np.log(np.maximum(F, 1e-12)), N]; clf = {}
        for gen in ('py', 'hw'):
            Lt, yt, Nt = H14.efp_train_lists(gen, n_per_class=1000 if smoke else 15000, seed=SEED + 2)
            Ft = H14.efps(Lt, pool)
            clf[gen] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.15,
                                                      random_state=0).fit(feat(Ft, Nt), (yt == 0).astype(int))
            log(f'efpn_{gen} trained on {len(yt)} jets')
        for gen in ('py', 'hw'):
            L0, y, inf = load(gen, a.per_class); meta, P = make_edits(L0, gen)
            rs = np.random.default_rng(SEED + 5); nsub = min(500, a.per_class)
            sub = np.sort(np.concatenate([rs.choice(np.flatnonzero(y == c), nsub, replace=False) for c in (0, 1)]))
            Ls = [L0[i] for i in sub]; out = dict(y=y, sub=sub)
            for c in [c for c in conds(P) if c[0] in 'bmc']:
                Le = Ls if c == 'base' else apply(Ls, meta['sel'][sub], P[c][sub])
                F = H14.efps([x[:, :3] for x in Le], pool); N = np.array([len(x) for x in Le], float)
                for tg in ('py', 'hw'):
                    v = np.full(len(y), np.nan, np.float32); v[sub] = clf[tg].decision_function(feat(F, N)); out[f'{c}|efpn_{tg}'] = v
                if c[0] == 'm': log(gen, c, 'efp done')
            np.savez_compressed(RES / f'h30_efp_{gen}{tag}.npz', **out); log('wrote', RES / f'h30_efp_{gen}{tag}.npz')
    pool.close(); log('ref step done')


def step_edchk(a):
    """the CPU environment rebuilds the same edits as the GPU eval step (the ref step ran before the eval edits file existed)"""
    for gen in ('py', 'hw'):
        L0, y, inf = load(gen, a.per_class); meta, P = make_edits(L0, gen); Z = np.load(RES / f'h30_edits_{gen}.npz')
        assert (Z['sel'] == meta['sel']).all() and (Z['y'] == y).all()
        log(gen, 'max |param difference| eval vs CPU rebuild', max(float(np.abs(Z[f'prm|{c}'] - P[c]).max()) for c in P))


# -------------------------------------------------------------------------------------------------------------- statistics
def _load(gen, tag):
    Z = np.load(RES / f'h30_logits_{gen}{tag}.npz'); y = Z['y'].astype(int); S, ok, obs, X = {}, {}, {}, {}
    for k in Z.files:
        p = k.split('|')
        if p[0] == 'ok': ok[p[1]] = Z[k].astype(bool)
        elif p[0] == 'obs': obs[(p[1], p[2])] = Z[k].astype(float)
        elif p[0] in ('R', 'kt'): X[(p[0], p[1])] = Z[k].astype(float)
        elif len(p) == 2: S[(p[0], p[1])] = Z[k].astype(float)
    meta = {k: Z[k] for k in ('sel', 'dR0', 'kt0', 'z0', 'pTJ')}
    for f in (f'h30_ref_{gen}{tag}.npz', f'h30_efp_{gen}{tag}.npz'):
        if not (RES / f).exists(): log('missing', f); continue
        Zb = np.load(RES / f); assert (Zb['y'] == y).all()
        for k in Zb.files:
            p = k.split('|')
            if len(p) == 2: S[(p[0], p[1])] = Zb[k].astype(float)
    return y, S, ok, obs, X, meta


def _fit(Y, X):
    """OLS of cell means Y (..., C) on standardized regressors X (..., C, p) with intercept -> coefs (..., p), R2"""
    Xs = (X - X.mean(-2, keepdims=True)) / X.std(-2, keepdims=True)
    A = np.concatenate([np.ones(Xs.shape[:-1] + (1,)), Xs], -1)
    beta = np.linalg.lstsq(A, Y, rcond=None)[0] if A.ndim == 2 else np.stack([np.linalg.lstsq(A[b], Y[b], rcond=None)[0] for b in range(len(A))])
    res = Y - (A @ beta[..., None])[..., 0] if A.ndim == 3 else Y - A @ beta
    r2 = 1 - (res ** 2).sum(-1) / ((Y - Y.mean(-1, keepdims=True)) ** 2).sum(-1)
    return beta[..., 1:], r2


def bh(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); q = np.empty(m); run = 1.0
    for r in range(m - 1, -1, -1): run = min(run, ps[o[r]] * m / (r + 1)); q[o[r]] = run
    return q


def step_stats(a):
    from interp import h5_qg_counting as H
    from interp.h14_qg_generator import Boot
    tag = '_smoke' if a.per_class < 2000 else ''
    Ld = np.load(RES / f'h30_lund{tag}.npz'); Tt = {g: Ld[f'T_{g}'] for g in ('py', 'hw')}
    R = dict(sample=dict(per_class=a.per_class, seed=SEED), grid=dict(dR_centres=RC.tolist(), kt_centres_GeV=KC.tolist(),
             dR_edges=RE.tolist(), kt_edges=KE.tolist(), zcap=ZCAP), models_info=json.loads((RES / f'h30_models{tag}.json').read_text()),
             lund=json.loads((RES / f'h30_lund{tag}.json').read_text()) if (RES / f'h30_lund{tag}.json').exists() else None,
             rho_ratio_grid={g: np.log(Ld[f'rho_0_{g}'] / np.maximum(Ld[f'rho_1_{g}'], 1e-9)).tolist() for g in ('py', 'hw')}, gens={})
    holm_tests = []
    for gen in ('py', 'hw'):
        y, S, ok, obs, X, meta = _load(gen, tag); J = len(y); Bt = Boot(J, B=1000, P=2000, seed=SEED)
        cells = [(r, k) for r, k in CELLS if cn('m', r, k) in ok and ok[cn('m', r, k)].sum() >= min(200, J // 4)]
        scorers = sorted({m for (c, m) in S if c == 'base'})
        g = dict(J=J, cells=[[int(r), int(k)] for r, k in cells], unedited={}, maps={}, regress={}, fits={}, pairs={})
        for m in scorers:
            s0 = S[('base', m)]; o = ~np.isnan(s0)
            g['unedited'][m] = dict(**H.roc_metrics(s0[o], y[o]), n=int(o.sum()))
        # per-cell regressors (main - c1): delta ln(rho_g/rho_q) table lookup (own and other generator), girth, pT^D
        src = {tg: Tt[tg][fbin(np.log(1 / np.maximum(meta['dR0'], 1e-6)), np.log(np.maximum(meta['kt0'], 1e-6)))] for tg in Tt}
        reg = {}
        for (r, k) in cells:
            cm, cc = cn('m', r, k), cn('c', r, k); o = ok[cm] & ok[cc]
            dst = {tg: Tt[tg][fbin(np.log(1 / np.maximum(X[('R', cm)], 1e-6)), np.log(np.maximum(X[('kt', cm)], 1e-6)))] for tg in Tt}
            rv = dict(lund_py=dst['py'] - src['py'], lund_hw=dst['hw'] - src['hw'],
                      girth=obs[(cm, 'girth')] - obs[(cc, 'girth')], ptd=obs[(cm, 'ptd')] - obs[(cc, 'ptd')])
            reg[(r, k)] = (rv, o)
            g['regress'][f'{r}{k}'] = dict(n=int(o.sum()), dR=float(X[('R', cm)][o].mean()), kt=float(X[('kt', cm)][o].mean()),
                                           **{kk: float(v[o].mean()) for kk, v in rv.items()},
                                           c1_dgirth=float((obs[(cc, 'girth')] - obs[('base', 'girth')])[o].mean()),
                                           c1_dR=float(X[('R', cc)][o].mean() - meta['dR0'][o].mean()))
        regs = ['lund', 'girth', 'ptd']
        Xc = {}; Xb = {}
        for tg in ('py', 'hw'):
            Xc[tg] = np.array([[reg[c][0][f'lund_{tg}' if q == 'lund' else q][reg[c][1]].mean() for q in regs] for c in cells])
            Xb[tg] = np.stack([np.stack([Bt.mean_ci(reg[c][0][f'lund_{tg}' if q == 'lund' else q], reg[c][1])[1] for q in regs], -1)
                               for c in cells], 1)                                       # (B, C, 3)
        g['regressor_corr'] = {tg: np.corrcoef(Xc[tg].T).tolist() for tg in Xc}
        D = {}
        for m in scorers:
            mp = {}; pv = []; Yb = []; Ym = []
            for (r, k) in cells:
                cm, cc, cr = cn('m', r, k), cn('c', r, k), cn('r', r, k)
                if (cm, m) not in S: continue
                b0 = S[('base', m)]; o = ok[cm] & ok[cc] & ~np.isnan(b0)
                dm, dc = S[(cm, m)] - b0, S[(cc, m)] - b0; d = dm - dc
                d = np.where(o, d, 0); ci, bs = Bt.mean_ci(d, o); p = Bt.signflip(d, o)
                e = dict(n=int(o.sum()), main=float(dm[o].mean()), c1=float(dc[o].mean()), c1_ci=Bt.mean_ci(np.where(o, dc, 0), o)[0],
                         diff=float(d[o].mean()), diff_ci=ci, p=p,
                         by_class=dict(q=float(d[o & (y == 1)].mean()), g=float(d[o & (y == 0)].mean())))
                if (cr, m) in S:
                    o2 = o & ok[cr]; d2 = np.where(o2, S[(cr, m)] - b0, 0)
                    e.update(c2=float(d2[o2].mean()), c2_ci=Bt.mean_ci(d2, o2)[0], main_minus_c2=float((dm - d2)[o2].mean()))
                mp[f'{r}{k}'] = e; pv.append(p); Yb.append(bs); Ym.append(e['diff']); D[(m, (r, k))] = (d, o)
            for key, q in zip(mp, bh(pv)): mp[key]['bh'] = float(q)
            g['maps'][m] = mp
            if len(Ym) != len(cells): continue
            Ym = np.array(Ym); Yb = np.stack(Yb, 1); tg = TRAINED.get(m, m.split('_')[-1] if m.split('_')[-1] in ('py', 'hw') else gen)
            fits = {}
            for tl in ('py', 'hw'):
                beta, r2 = _fit(Ym, Xc[tl]); bb, _ = _fit(Yb, Xb[tl])
                rp = np.random.default_rng(SEED + 11); cnt = np.zeros(3); NP = 5000
                for _ in range(NP):
                    for q in range(3):
                        Xp = Xc[tl].copy(); Xp[:, q] = Xp[rp.permutation(len(cells)), q]
                        cnt[q] += abs(_fit(Ym, Xp)[0][q]) >= abs(beta[q]) - 1e-12
                single = {q: float(_fit(Ym, Xc[tl][:, [i]])[1]) for i, q in enumerate(regs)}
                fits[tl] = dict(coef={q: float(beta[i]) for i, q in enumerate(regs)},
                                coef_ci={q: [float(np.percentile(bb[:, i], 2.5)), float(np.percentile(bb[:, i], 97.5))] for i, q in enumerate(regs)},
                                p_perm={q: float((cnt[i] + 1) / (NP + 1)) for i, q in enumerate(regs)}, R2=float(r2), R2_single=single,
                                map_sd=float(Ym.std()))
            g['fits'][m] = dict(own_table=tg, **fits)
            for q in regs: holm_tests.append((gen, m, q, fits[tg]['p_perm'][q]))
            f = fits[tg]
            log(gen, m, f"AUC {g['unedited'][m]['AUC']:.4f} acc {g['unedited'][m]['acc']:.4f} | map sd {Ym.std():.4f} range "
                f"[{Ym.min():+.4f}, {Ym.max():+.4f}] | fit({tg}) a {f['coef']['lund']:+.4f} b {f['coef']['girth']:+.4f} "
                f"c {f['coef']['ptd']:+.4f} R2 {f['R2']:.2f} single {', '.join(f'{k} {v:.2f}' for k, v in f['R2_single'].items())}")
        # Herwig-trained minus Pythia-trained, paired on the same jets, per cell; low-kt (< 1 GeV) vs high-kt contrast
        dT = np.array([np.mean((reg[c][0]['lund_hw'] - reg[c][0]['lund_py'])[reg[c][1]]) for c in cells])
        low = np.array([KC[k] < 1.0 for r, k in cells])
        for mh, mp_ in PAIRS:
            if mh not in g['maps'] or mp_ not in g['maps']: continue
            e = {}; pv = []; mb = []
            for c in cells:
                (dh, o), (dp, _) = D[(mh, c)], D[(mp_, c)]; dd = np.where(o, dh - dp, 0); ci, bs = Bt.mean_ci(dd, o); p = Bt.signflip(dd, o)
                e[f'{c[0]}{c[1]}'] = dict(diff=float(dd[o].mean()), ci=ci, p=p); pv.append(p); mb.append(bs)
            for key, q in zip(e, bh(pv)): e[key]['bh'] = float(q)
            mv = np.array([v['diff'] for v in e.values()]); mb = np.stack(mb, 1)
            con = np.abs(mv[low]).mean() - np.abs(mv[~low]).mean(); conb = np.abs(mb[:, low]).mean(1) - np.abs(mb[:, ~low]).mean(1)
            rp = np.random.default_rng(SEED + 13); perm = np.array([(lambda s: np.abs(mv[s]).mean() - np.abs(mv[~s]).mean())(rp.permutation(low)) for _ in range(5000)])
            r_dT = float(np.corrcoef(mv, dT)[0, 1])
            rperm = np.array([np.corrcoef(mv[rp.permutation(len(mv))], dT)[0, 1] for _ in range(5000)])
            g['pairs'][f'{mh} - {mp_}'] = dict(cells=e, n_bh005=int(sum(v['bh'] < 0.05 for v in e.values())),
                                                mean_abs_low_kt=float(np.abs(mv[low]).mean()), mean_abs_high_kt=float(np.abs(mv[~low]).mean()),
                                                contrast=float(con), contrast_ci=[float(np.percentile(conb, 2.5)), float(np.percentile(conb, 97.5))],
                                                contrast_p_perm=float(((perm >= con).sum() + 1) / 5001),
                                                corr_with_dlnrho_hw_minus_py=r_dT, corr_p_perm=float(((np.abs(rperm) >= abs(r_dT)).sum() + 1) / 5001))
            holm_tests.append((gen, f'{mh} - {mp_}', 'low-kt contrast', g['pairs'][f'{mh} - {mp_}']['contrast_p_perm']))
            log(gen, mh, '-', mp_, f"cells BH<0.05 {g['pairs'][f'{mh} - {mp_}']['n_bh005']}/{len(cells)}; |diff| low kt {np.abs(mv[low]).mean():.4f} "
                f"high kt {np.abs(mv[~low]).mean():.4f} (p {g['pairs'][f'{mh} - {mp_}']['contrast_p_perm']:.3g}); corr with dln rho(hw-py) {r_dT:+.2f}")
        g['dlnrho_hw_minus_py_cells'] = dT.tolist()
        R['gens'][gen] = g
    # Holm over models within each (generator, coefficient) family
    fam = {}
    for t in holm_tests: fam.setdefault((t[0], t[2]), []).append(t)
    R['holm'] = []
    for key, ts in fam.items():
        adj = H.holm([t[3] for t in ts])
        R['holm'] += [dict(gen=t[0], model=t[1], test=t[2], p=t[3], holm=float(h)) for t, h in zip(ts, adj)]
    (RES / f'h30_run{tag}.json').write_text(json.dumps(R, indent=1, default=float)); log('wrote', RES / f'h30_run{tag}.json')


# ----------------------------------------------------------------------------------------------------------------- figure
def step_fig(a):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    tag = '_smoke' if a.per_class < 2000 else ''
    R = json.loads((RES / f'h30_run{tag}.json').read_text()); gen = 'py'; G = R['gens'][gen]
    panels = [('part_kin', 'ParT_kin, Pythia-trained'), ('hw_part_kin', 'ParT_kin, Herwig-trained'),
              ('pnet_kin', 'ParticleNet_kin, Pythia-trained'), ('efn_k', 'EFN (IRC-safe), Pythia-trained'),
              ('lund_py', 'LR(primary Lund density), Pythia'), ('lrg_py', 'LR(girth), Pythia'), ('lrp_py', 'LR($p_T^D$), Pythia'),
              ('part_kin_perm', 'ParT_kin, permuted weights (null)')]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax, (m, t) in zip(axs.flat, panels):
        Z = np.full((6, 6), np.nan); Sg = np.zeros((6, 6), bool)
        for key, e in G['maps'].get(m, {}).items():
            r, k = int(key[0]), int(key[1]); Z[r, k] = e['diff']; Sg[r, k] = e['bh'] < 0.05
        v = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
        im = ax.pcolormesh(RE, KE, Z.T, cmap='RdBu_r', vmin=-v, vmax=v, shading='flat')
        for r in range(6):
            for k in range(6):
                if np.isfinite(Z[r, k]):
                    ax.text(RC[r], KC[k], f'{Z[r, k]:+.2f}' + ('' if Sg[r, k] else ' ns'), ha='center', va='center', fontsize=6.5,
                            color='k' if abs(Z[r, k]) < 0.6 * v else 'w')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis(); ax.set_title(t, fontsize=10)
        ax.set_xticks([0.3, 0.2, 0.1, 0.05, 0.03]); ax.set_xticklabels(['0.3', '0.2', '0.1', '0.05', '0.03'])
        ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(['0.5', '1', '2', '5']); ax.minorticks_off()
        ax.set_xlabel('ΔR of moved particle to jet axis'); ax.set_ylabel('$k_t = p_{T,i}\\,ΔR$ [GeV]')
        cb = fig.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=7)
    fig.suptitle(f'Change in gluon log-odds when one soft particle (0.5–2 % of jet $p_T$) is moved to the cell, minus a rotation of '
                 f'the same particle at its own ΔR ({R["sample"]["per_class"]} quark + {R["sample"]["per_class"]} gluon Pythia jets; '
                 f'grey = not computed; "ns" = BH q ≥ 0.05; colour scale per panel)', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / f'h30_lund_map{tag}.png', dpi=120); log('wrote h30_lund_map.png')
    # coefficient figure
    ms = [m for m in ('part_kin', 'pnet_kin', 'hw_part_kin', 'hw_pnet_kin', 'pfn_k', 'pfn_t', 'efn_k', 'part_kin_perm',
                      'lund_py', 'lund_hw', 'lrg_py', 'lrp_py', 'efpn_py') if m in G['fits']]
    fig, axs = plt.subplots(1, 2, figsize=(13, 0.4 * len(ms) + 2), sharey=True)
    for ax, gen in zip(axs, ('py', 'hw')):
        F = R['gens'][gen]['fits']
        for q, (nm, c, dy) in enumerate((('lund', '#1b6ca8', -0.22), ('girth', '#c2185b', 0), ('ptd', '#6a8d00', 0.22))):
            mu = np.array([F[m][F[m]['own_table']]['coef'][nm] for m in ms]); ci = np.array([F[m][F[m]['own_table']]['coef_ci'][nm] for m in ms])
            ax.errorbar(mu, np.arange(len(ms)) + dy, xerr=[mu - ci[:, 0], ci[:, 1] - mu], fmt='o', ms=4, color=c, lw=1,
                        label={'lund': 'Δ ln ρ_g/ρ_q (a)', 'girth': 'Δ girth (b)', 'ptd': 'Δ $p_T^D$ (c)'}[nm])
        ax.axvline(0, color='k', lw=0.6); ax.set_title(f'{"Pythia" if gen == "py" else "Herwig"} test jets', fontsize=10)
        ax.set_xlabel('coefficient per 1 SD of the regressor across cells [Δ log-odds] (95 % CI)')
    lab = dict(part_kin='ParT_kin [Py-trained]', pnet_kin='ParticleNet_kin [Py]', hw_part_kin='ParT_kin [Hw-trained]', hw_pnet_kin='ParticleNet_kin [Hw]',
               pfn_k='PFN [Py]', pfn_t='PFN, 2nd run [Py]', efn_k='EFN, IRC-safe [Py]', part_kin_perm='ParT_kin, permuted weights',
               lund_py='LR(Lund density), Py', lund_hw='LR(Lund density), Hw', lrg_py='LR(girth), Py', lrp_py='LR($p_T^D$), Py', efpn_py='GBDT(EFP+N), Py')
    axs[0].set_yticks(range(len(ms))); axs[0].set_yticklabels([lab[m] for m in ms], fontsize=8.5); axs[0].invert_yaxis(); axs[0].legend(fontsize=8)
    fig.suptitle('Fit of the cell map (main − rotation) on the cell means of three regressors', fontsize=10)
    fig.tight_layout(); fig.savefig(RES / f'h30_lund_fit{tag}.png', dpi=120); log('wrote h30_lund_fit.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True, choices=['eval', 'ref', 'stats', 'fig', 'edchk'])
    ap.add_argument('--gens', default='py,hw'); ap.add_argument('--per_class', type=int, default=2000)
    ap.add_argument('--cpu', type=int, default=4); ap.add_argument('--part', default='lund', choices=['lund', 'efp'])
    a = ap.parse_args()
    dict(eval=step_eval, ref=step_ref, stats=step_stats, fig=step_fig, edchk=step_edchk)[a.step](a)
