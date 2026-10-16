"""P1 — the translation test on ParT (mirror of JEDI E6; claims A4 / A8; stage 4).

JEDI-linear's mass variable is √(M0·M2) about the given axis (M0 = Σ pT, M2 = Σ pT r²): a rigid translation of all particles
by 0.06 rad, which leaves the invariant mass unchanged, moved its read-out mass by 0.24–0.29 in log and turned 76 % of W calls
into Z (inbox/E6_mass.md). ParT's pair features (ln kT, ln z, ln ΔR, ln m²) are translation invariant by construction; its
per-particle inputs Δη, Δφ, ΔR (and, under an η shift, e_log and logerel) are not.

Sample: 1,000 test jets per class (rows 3,000–4,000 of each class's first test file); ParT_full and ParT_kin.
Edit: every real particle's (Δη, Δφ) shifted by δ ∈ {0.01, 0.02, 0.03, 0.045, 0.06, 0.09} along η or along φ; pT and particle
masses kept; px, py, pz, E rebuilt with part_core.build (eta = jet_eta + sign(jet_eta)·Δη); padded slots stay padded. The φ shift
is an exact rotation (m unchanged); the η shift at fixed pT and particle mass is a rapidity shift up to the particle masses
(Δlog m measured with jmass). Baseline: the same jets rebuilt with δ = 0.
Controls: (a) JEDI's numbers from inbox/E6_mass.md (plotted by p1_plot.py); (b) angles × 1.1 and × 1.25 on the same jets
(mass changes; G192: W moved to 91 GeV → Z 37 %).
Metrics per δ: (i) accuracy, fraction of calls changed; (ii) W→Z, Z→W (fraction of true W / Z jets called Z / W), mean
Δ(logit_W − logit_Z) on W jets; (iii) shift of the W/Z neuron n98 (index 97) in units of its spread, all jets / W / Z, and of
every active neuron; (iv) a ridge read-out of log m fitted on the unedited class token z (fixed 50/50 split, alpha on a
validation part of the train half; held-out R²; linear, and linear + products of the top-8 PCA components as in E6) and its mean
shift on the test half under each edit, against the √(M0M2) prediction ½ Δlog(M0·M2) from the edited particles and against 0;
the across-jet slope of Δlog m̂ on the per-jet prediction.
  ./run_part.sh interp/p1_translation_part.py [per_class] > logs/p1_translation.log 2>&1  -> results/part/p1_translation.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.part_core import ParTNetwork, sample, kin, build, jmass, moments, OUT, CL, N98, ACTIVE

t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(*a, flush=True)
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DELTAS = (0.01, 0.02, 0.03, 0.045, 0.06, 0.09)
SCALES = (1.1, 1.25)
SEED = 1
iW, iZ = CL.index('Wqq'), CL.index('Zqq')

J0 = sample(PER, 3000); y = J0['label'].astype(int); N = len(y)
pt, de, dp, ms = kin(J0); real = pt > 0
Jb = build(J0, pt, de, dp, ms)                                   # baseline: rebuilt, no edit
m0 = jmass(Jb); M0m = moments(*kin(Jb)[:3]); logm0 = np.log(np.maximum(m0, 1.0))
log(f'{N} jets ({PER}/class, rows 3000-{3000 + PER}); mass median {np.median(m0):.1f} GeV; |M1|^2/(M0M2) median {np.median(M0m["M1"] ** 2 / M0m["M0M2"]):.2e}; '
    f'M2/M0 median {np.median(M0m["M2"] / M0m["M0"]):.4f} ({time.time() - t0:.0f} s)')

EDITS = {}
for d in DELTAS:
    EDITS[f'eta{d}'] = build(J0, pt, de + d * real, dp, ms)
    EDITS[f'phi{d}'] = build(J0, pt, de, dp + d * real, ms)
for k in SCALES:
    EDITS[f'k{k}'] = build(J0, pt, de * k, dp * k, ms)
truth = {}
for name, Je in EDITS.items():
    pe, dee, dpe, _ = kin(Je); Me = moments(pe, dee, dpe); me = jmass(Je)
    truth[name] = dict(dlogm=np.log(np.maximum(me, 1.0)) - logm0, dlog_sqrtM0M2=0.5 * np.log(Me['M0M2'] / M0m['M0M2']),
                       dlog_sqrtM2=0.5 * np.log(Me['M2'] / M0m['M2']), dlogM0=np.log(Me['M0'] / M0m['M0']),
                       padded_ok=bool(np.all((Je['mask'] > 0) == (Jb['mask'] > 0))) and bool(np.all(Je['deta'][Je['mask'] == 0] == 0) and np.all(Je['dphi'][Je['mask'] == 0] == 0)),
                       pt_ok=float(np.abs(np.hypot(Je['px'], Je['py']) - np.hypot(Jb['px'], Jb['py'])).max()))
log('edits: name | Δlog m (mean, max |.|) | ½Δlog(M0M2) mean | Δlog M0 | padded slots unchanged | max |ΔpT|')
for name, t in truth.items():
    log(f'   {name:8s} | {t["dlogm"].mean():+.5f} {np.abs(t["dlogm"]).max():.4f} | {t["dlog_sqrtM0M2"].mean():+.4f} | {np.abs(t["dlogM0"]).max():.1e} | {t["padded_ok"]} | {t["pt_ok"]:.1e}')

# ---------------------------------------------------------------- read-out (as interp/e6_mass_jedi.py)
rng = np.random.default_rng(SEED); perm = rng.permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


class Ridge:
    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-9] = np.inf
        Z = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std()
        tt = (t - self.tm) / self.ts; best = None
        for a in ALPHAS:
            w = self._fit(Z[idx_fit], tt[idx_fit], a); e = ((Z[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = best[1]; self.w = self._fit(Z[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Z, t, a):
        Zc = Z - Z.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Z) * np.eye(Z.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


class PCAProd:
    def __init__(self, X, idx, K=8):
        self.mu = X[idx].mean(0); sd = X[idx].std(0); sd[sd < 1e-9] = np.inf; self.sd = sd
        Z = (X[idx] - self.mu) / self.sd; _, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        self.K = min(K, int((s > 1e-8).sum())); self.V = Vt[:self.K].T; self.iu = np.triu_indices(self.K)

    def __call__(self, X):
        C = ((X - self.mu) / self.sd) @ self.V; return np.concatenate([X, (C[:, :, None] * C[:, None, :])[:, self.iu[0], self.iu[1]]], 1)


r2 = lambda p, t: float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())
res = dict(N=N, per_class=PER, rows=[3000, 3000 + PER], seed=SEED, deltas=list(DELTAS), scales=list(SCALES), n98_index=N98, active=ACTIVE,
           mass_median=float(np.median(m0)), M2_over_M0_median=float(np.median(M0m['M2'] / M0m['M0'])),
           truth={k: dict(dlogm_mean=float(v['dlogm'].mean()), dlogm_absmax=float(np.abs(v['dlogm']).max()), dlogm_te=float(v['dlogm'][te].mean()),
                          dlog_sqrtM0M2=float(v['dlog_sqrtM0M2'].mean()), dlog_sqrtM0M2_te=float(v['dlog_sqrtM0M2'][te].mean()),
                          dlog_sqrtM0M2_W_te=float(v['dlog_sqrtM0M2'][te][y[te] == iW].mean()), dlog_sqrtM2=float(v['dlog_sqrtM2'].mean()),
                          padded_ok=v['padded_ok']) for k, v in truth.items()},
           models={})
for n in ('full', 'kin'):
    net = ParTNetwork(n); t1 = time.time()
    Rraw = net.run(J0); craw = Rraw['logits'].argmax(1)
    R0 = net.run(Jb); z0, L0 = R0['z'], R0['logits']; c0 = L0.argmax(1); sd0 = z0.std(0)
    W, b = net.last[0], net.last[1]
    row = dict(acc_raw=float((craw == y).mean()), acc=float((c0 == y).mean()), calls_changed_by_rebuild=float((c0 != craw).mean()),
               n98=dict(sd=float(sd0[N98]), w_W_per_sd=float(W[N98, iW] * sd0[N98]), w_Z_per_sd=float(W[N98, iZ] * sd0[N98]),
                        corr_logm=float(np.corrcoef(z0[:, N98], logm0)[0, 1])),
               active_sd={int(u): float(sd0[u]) for u in ACTIVE}, n_active=int((sd0 > 0.01).sum()),
               own_class=[float((c0[y == k] == k).mean()) for k in range(10)])
    log(f'ParT_{n}: accuracy {100 * row["acc"]:.1f} % (raw jets {100 * row["acc_raw"]:.1f} %, rebuild changes {100 * row["calls_changed_by_rebuild"]:.2f} % of calls); '
        f'n98 (index {N98}): sd {row["n98"]["sd"]:.3f}, last layer × sd: W {row["n98"]["w_W_per_sd"]:+.2f} Z {row["n98"]["w_Z_per_sd"]:+.2f}, corr with log m {row["n98"]["corr_logm"]:+.3f}; '
        f'{row["n_active"]} neurons with sd > 0.01 ({time.time() - t1:.0f} s)')
    # read-outs on the unedited class token
    lin = Ridge(z0, logm0, trf, trv, tr); pp = PCAProd(z0, tr); nl = Ridge(pp(z0), logm0, trf, trv, tr)
    linM2 = Ridge(z0, np.log(M0m['M2']), trf, trv, tr)
    row['readout'] = dict(logm_lin_r2=r2(lin(z0)[te], logm0[te]), logm_nl_r2=r2(nl(pp(z0))[te], logm0[te]), alpha_lin=float(lin.alpha), alpha_nl=float(nl.alpha),
                          logM2_lin_r2=r2(linM2(z0)[te], np.log(M0m['M2'])[te]), logm_lin_r2_active_only=r2(Ridge(z0[:, ACTIVE], logm0, trf, trv, tr)(z0[:, ACTIVE])[te], logm0[te]))
    log('   read-out of log m from z (held-out R²): linear {logm_lin_r2:.3f} (13 active only {logm_lin_r2_active_only:.3f}), + PCA products {logm_nl_r2:.3f}; log M2 linear {logM2_lin_r2:.3f}'.format(**row['readout']))
    base_lin, base_nl = lin(z0[te]), nl(pp(z0[te]))
    row['edits'] = {}
    log('   edit     | acc  changed | W→Z  Z→W  W→W  Z→Z  Δ(lW−lZ)|W | n98 shift/sd all  W     Z    | Δlog m̂ lin  nl   | pred ½ΔlogM0M2  Δlog m | slope r')
    for name, Je in EDITS.items():
        R = net.run(Je); z, L = R['z'], R['logits']; c = L.argmax(1); t = truth[name]
        dz = (z - z0) / np.where(sd0 > 1e-9, sd0, np.inf)
        dl = lin(z[te]) - base_lin; dn = nl(pp(z[te])) - base_nl; pr = t['dlog_sqrtM0M2'][te]
        prc = pr - pr.mean(); slope = float((prc @ (dl - dl.mean())) / max(prc @ prc, 1e-30)); corr = float(np.corrcoef(pr, dl)[0, 1]) if pr.std() > 1e-9 else None
        e = dict(acc=float((c == y).mean()), changed=float((c != c0).mean()),
                 W_to_Z=float((c[y == iW] == iZ).mean()), Z_to_W=float((c[y == iZ] == iW).mean()), W_to_W=float((c[y == iW] == iW).mean()), Z_to_Z=float((c[y == iZ] == iZ).mean()),
                 Wcalls_to_Z=float((c[(c0 == iW) & (y == iW)] == iZ).mean()),
                 dlogit_WZ_on_W=float(((L[:, iW] - L[:, iZ]) - (L0[:, iW] - L0[:, iZ]))[y == iW].mean()),
                 n98_shift=float(dz[:, N98].mean()), n98_shift_W=float(dz[y == iW, N98].mean()), n98_shift_Z=float(dz[y == iZ, N98].mean()), n98_abs=float(np.abs(dz[:, N98]).mean()),
                 active_shift={int(u): float(dz[:, u].mean()) for u in ACTIVE}, active_abs={int(u): float(np.abs(dz[:, u]).mean()) for u in ACTIVE},
                 dlogm_hat_lin=float(dl.mean()), dlogm_hat_nl=float(dn.mean()), dlogm_hat_lin_W=float(dl[y[te] == iW].mean()),
                 pred_sqrtM0M2=float(pr.mean()), dlogm_true=float(t['dlogm'][te].mean()), slope_on_pred=slope, corr_on_pred=corr,
                 own_class=[float((c[y == k] == k).mean()) for k in range(10)], calls_from_W=[float((c[y == iW] == k).mean()) for k in range(10)])
        row['edits'][name] = e
        log(f'   {name:8s} | {100 * e["acc"]:4.1f} {100 * e["changed"]:5.1f} % | {e["W_to_Z"]:.3f} {e["Z_to_W"]:.3f} {e["W_to_W"]:.3f} {e["Z_to_Z"]:.3f} {e["dlogit_WZ_on_W"]:+6.2f} | '
            f'{e["n98_shift"]:+.3f} {e["n98_shift_W"]:+.3f} {e["n98_shift_Z"]:+.3f} | {e["dlogm_hat_lin"]:+.4f} {e["dlogm_hat_nl"]:+.4f} | {e["pred_sqrtM0M2"]:+.4f} {e["dlogm_true"]:+.5f} | '
            f'{slope:+.2f} {corr if corr is None else round(corr, 2)}')
    e0 = dict(W_to_Z=float((c0[y == iW] == iZ).mean()), Z_to_W=float((c0[y == iZ] == iW).mean()), W_to_W=float((c0[y == iW] == iW).mean()), Z_to_Z=float((c0[y == iZ] == iZ).mean()))
    row['unedited'] = e0; log(f'   unedited | W→Z {e0["W_to_Z"]:.3f} Z→W {e0["Z_to_W"]:.3f} W→W {e0["W_to_W"]:.3f} Z→Z {e0["Z_to_Z"]:.3f}')
    big = sorted(ACTIVE, key=lambda u: -abs(row['edits']['phi0.06']['active_shift'][u]))[:5]
    log('   largest mean neuron shifts at phi0.06 (units of sd): ' + ', '.join(f'n{u + 1}(idx {u}) {row["edits"]["phi0.06"]["active_shift"][u]:+.2f}' for u in big)
        + ' | at k1.1: ' + ', '.join(f'idx {u} {row["edits"]["k1.1"]["active_shift"][u]:+.2f}' for u in sorted(ACTIVE, key=lambda u: -abs(row['edits']['k1.1']['active_shift'][u]))[:5]))
    res['models'][n] = row
    del net
    try:
        import torch; torch.mps.empty_cache()
    except Exception:
        pass
json.dump(res, open(OUT / 'p1_translation.json', 'w'), indent=1); log(f'P1 DONE ({time.time() - t0:.0f} s)')
