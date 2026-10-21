"""H48 verification (loop D, verifier). Fresh-sample rerun of the H48 lepton-endpoint edit plus two controls that H48 lacks.

Question: is the extra drop that H48 found for H->l nu q q' jets when the lepton edit pushes the visible mass across 125 GeV
(i) reproduced on a fresh sample, (ii) localized at the edge (a step at 125 GeV) rather than a conditional mass response that is
just as steep above the edge, and (iii) present when the mass is raised by moving a hadron instead of the lepton (so the lepton's
isolation and the lepton-quark angles are not what changes)?

Sample (random rows, never first-k): JetClass test_20M *_101 files (H48 used *_100): TTBarLep 6000, HToWW2Q1L 8000,
  ZJetsToNuNu 2000, TTBar 2000 random rows (np.random.default_rng(4851 + k)). Every jet rebuilt about its own E-scheme axis
  (h48.rebuild, part_core.build convention eta = jet_eta + sign(jet_eta) * deta).
Edits (one particle p moves; its energy, |p|, mass, PID and impact parameters kept; every other particle untouched):
  E_lep / C_lep  as H48: p = the lepton (hardest isElectron/isMuon particle); E rotates p in the (p_rest, p_p) plane to the
                 opening angle that changes the jet mass by exactly dm; C rotates p about the p_rest direction by the azimuth
                 that moves it by the same 3-D angle (theta and jet mass fixed).
  E_had / C_had  NEW: p = the hardest particle not flagged as a lepton (a hadron or photon of the b or q q' system); same E and C
                 construction. The lepton does not move, so its isolation and its angles to the quark particles change only
                 through the moved hadron and the axis.
Windows (scan): unedited jet mass in [X + L, X + L + 10] GeV, L in {-63, -53, ..., +27}, dm = +20 GeV (X = 125 GeV for
  HToWW2Q1L scored with s_Hqql; X = 152.6 GeV for TTBarLep scored with s_Tbl). At most CAP jets per window (random subset).
  H48's arms: below = L in {-43, -33}; across = L = -13. Reverse arm (exploratory, as H48): [X + 7, X + 27], dm = -20.
Metric per jet: D = (E - C)_net - (E - C)_MLP(m), the network's response to the mass shift beyond what a mass-only classifier
  trained on natural jets predicts. Edge-localization contrasts on D: cross (L = -13) minus fully-below (L = -43, -33), and
  cross minus fully-above (L = +7, +17: start and end both above the edge).
Models: ParT_full, PN_full, ParT_kin, PN_kin; ParT_perm1, ParT_perm2 (null). References LR(m), MLP(m), MLP(m, lep) as H48 but
  trained on train_100M *_001 files (15,000 random rows per class, rng 4861..4864).
Statistics: bootstrap B = 1000 over jets; sign-flip permutation (5,000) for one-sample means; label permutation (2,000) for
  two-group contrasts; Holm across the family.
Run: ./sync.sh && AI_PY=conda ./job.sh d-h48-verify 6 12 'export PATH=/j-jepa-vol/aaron-interp/env-part/bin:$PATH
     JETDISTILL_TAGGER=part JETDISTILL_DEVICE=cpu JETDISTILL_DATA=/j-jepa-vol/aaron-interp/code/cluster/data_part
     JETDISTILL_MODELS=/j-jepa-vol/aaron-interp/models/part; python -m interp.h48_verify'
Outputs: $AI_RESULTS/physics/h48_verify.json, h48_verify_perjet.npz, h48_verify_scan.png
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import interp.h48_lepton_endpoint as H  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
SMOKE = bool(os.environ.get('H48V_SMOKE'))
RES = Path(os.environ.get('AI_RESULTS', 'results')) / 'physics'
TAG = 'h48_verify_smoke' if SMOKE else 'h48_verify'
NPER = dict(Tbl=6000, Hqql=8000, QCD=2000, Tbqq=2000)
if SMOKE: NPER = {k: 300 for k in NPER}
NREF = 2000 if SMOKE else 15000
CAP = 600
B = 1000
NPERM = 5000
LS = (-63, -53, -43, -33, -23, -13, -3, 7, 17, 27)
NETS = ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin', 'ParT_perm1', 'ParT_perm2')
TESTED = NETS[:4]
REFS = ('LR_m', 'MLP_m', 'MLP_mlep')
EDITS = ('lep', 'had')
CLS = H.CLS; SIGC = H.SIGC; EDGE = H.EDGE


def hadron_index(J):
    m = J['mask'] > 0; isl = (J['isElectron'] > 0) | (J['isMuon'] > 0); pt = np.hypot(J['px'], J['py'])
    key = np.where(m & ~isl, pt, -1.0); i = key.argmax(1); return np.where((m & ~isl).any(1), i, -1)


def edit_one(Ja, pi, dm, rng):
    """rotate particle slot pi (per jet) as in h48.edit_arm; returns E, C four-vectors, ok mask, bookkeeping"""
    P, m = H.p4(Ja); r = np.arange(len(pi)); assert (pi >= 0).all()
    Pp = P[r, pi].copy(); PR = P.sum(1) - Pp
    pp = np.linalg.norm(Pp[:, :3], axis=1); pr = np.linalg.norm(PR[:, :3], axis=1)
    rh = PR[:, :3] / pr[:, None]; ph = Pp[:, :3] / pp[:, None]
    cth = np.clip((ph * rh).sum(1), -1, 1); th = np.arccos(cth)
    mR2 = PR[:, 3] ** 2 - pr ** 2; mp2 = Pp[:, 3] ** 2 - pp ** 2; m0 = H.m4(P.sum(1)); m1 = m0 + dm
    cth1 = (mR2 + mp2 + 2 * PR[:, 3] * Pp[:, 3] - m1 ** 2) / (2 * pr * pp)
    okE = (np.abs(cth1) <= 1) & (m1 > 0) & (th > 1e-6); th1 = np.arccos(np.clip(cth1, -1, 1))
    eh = ph - cth[:, None] * rh; eh /= np.maximum(np.linalg.norm(eh, axis=1), 1e-12)[:, None]
    vE = np.cos(th1)[:, None] * rh + np.sin(th1)[:, None] * eh
    delta = np.abs(th1 - th); cphi = (np.cos(delta) - cth ** 2) / np.maximum(np.sin(th) ** 2, 1e-12)
    okC = okE & (cphi >= -1) & (cphi <= 1); phi = np.arccos(np.clip(cphi, -1, 1)) * rng.choice([-1.0, 1.0], len(pi))
    fh = np.cross(rh, eh)
    vC = cth[:, None] * rh + np.sin(th)[:, None] * (np.cos(phi)[:, None] * eh + np.sin(phi)[:, None] * fh)
    out = {}
    for v, lv in (('E', vE), ('C', vC)):
        Q = P.copy(); Q[r, pi, :3] = lv * pp[:, None]; Q[r, pi, 3] = Pp[:, 3]; out[v] = Q
    ok = okC.copy(); li = H.lepton_index(Ja)
    for v in ('E', 'C'):
        ae, ap = H.axis_of(out[v], m)
        for idx in (pi, li):      # moved particle and lepton both stay inside R = 0.8 of the edited axis
            Q = out[v][r, idx]; pt_ = np.hypot(Q[:, 0], Q[:, 1])
            dR = np.hypot(np.arcsinh(Q[:, 2] / pt_) - ae, H.wrap(np.arctan2(Q[:, 1], Q[:, 0]) - ap)); ok &= dR < 0.8
    mE = H.m4(out['E'].sum(1)); mC = H.m4(out['C'].sum(1))
    # energy and momentum magnitude of the moved particle conserved; others untouched
    for v in ('E', 'C'):
        assert np.allclose(np.linalg.norm(out[v][r, pi, :3], axis=1), pp) and np.allclose(out[v][r, pi, 3], Pp[:, 3])
        msk = np.ones(P.shape[:2], bool); msk[r, pi] = False; assert np.array_equal(out[v][msk], P[msk])
    if ok.any():
        assert np.allclose(mE[ok], m1[ok], atol=1e-3 * np.maximum(m1[ok], 1)), 'E mass target missed'
        assert np.allclose(mC[ok], m0[ok], atol=1e-3 * np.maximum(m0[ok], 1)), 'C changed the mass'
    return out, m, ok, dict(m0=m0, mE=mE, mC=mC, theta=th, theta_E=th1)


def boot(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0: return float('nan'), [float('nan')] * 2
    mm = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return float(x.mean()), np.percentile(mm, [2.5, 97.5]).tolist()


def signflip(x, rng):
    x = np.asarray(x, float)
    if len(x) < 2: return 1.0
    o = abs(x.mean()); s = rng.choice([-1.0, 1.0], (NPERM, len(x))); return float((np.sum(np.abs((s * x).mean(1)) >= o - 1e-12) + 1) / (NPERM + 1))


def contrast(a, b_, rng):
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    if len(a) < 5 or len(b_) < 5: return dict(mean=float('nan'), ci=[float('nan')] * 2, p_perm=1.0, n=[len(a), len(b_)])
    d = a.mean() - b_.mean(); bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b_[rng.integers(0, len(b_), (B, len(b_)))].mean(1)
    z = np.concatenate([a, b_]); cnt = 0; npm = 2000
    for _ in range(npm):
        q = rng.permutation(z); cnt += abs(q[:len(a)].mean() - q[len(a):].mean()) >= abs(d) - 1e-12
    return dict(mean=float(d), ci=np.percentile(bs, [2.5, 97.5]).tolist(), p_perm=float((cnt + 1) / (npm + 1)), n=[len(a), len(b_)])


def fit_refs(out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    F = {}; out['ref_rows'] = {}
    for k, (c, pre, lab) in enumerate(CLS):
        Jr, inf = H.load_rows(H.TRAIN / f'{pre}_001.root', NREF, 4861 + k); out['ref_rows'][pre] = inf
        P, m = H.p4(Jr); Jr = H.rebuild(Jr, P, m)       # same input path as the edited jets
        F[c] = H.obs(Jr); del Jr
    REF = {}; out['ref_info'] = {}
    for c in SIGC:
        Fc = {k: np.concatenate([F[x][k] for x in (c, 'QCD', 'Tbqq')]) for k in ('m', 'f', 'iso', 'has')}
        y = np.concatenate([np.ones(len(F[c]['m'])), np.zeros(len(F['QCD']['m']) + len(F['Tbqq']['m']))])
        tr = np.random.default_rng(4865).random(len(y)) < 0.8
        for r in REFS:
            X = np.column_stack([Fc[k] for k in H.REF_COLS[r]])
            mdl = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if r == 'LR_m' else
                   make_pipeline(StandardScaler(), MLPClassifier((32, 32), max_iter=400, early_stopping=True, random_state=1)))
            mdl.fit(X[tr], y[tr]); REF[(c, r)] = mdl
            out['ref_info'][f'{c}|{r}'] = dict(heldout_auc=H.auc(mdl.predict_proba(X[~tr])[:, 1], y[~tr].astype(int)))
            log(f'ref {c} {r}: ' + json.dumps(out['ref_info'][f'{c}|{r}']))
        grid = np.arange(40.0, 230.0, 5.0)
        out['ref_info'][f'{c}|MLP_m_curve'] = dict(m=grid, lo=H.lo(REF[(c, 'MLP_m')].predict_proba(grid[:, None])[:, 1]))
    return REF


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(node='H48_verify', smoke=SMOKE, nper=NPER, n_ref=NREF, cap=CAP, B=B, windows_L=LS, edges=EDGE)
    parts = []; out['sample'] = {}
    for k, (c, pre, lab) in enumerate(CLS):
        Jc, inf = H.load_rows(H.TEST / f'{pre}_101.root', NPER[c], 4851 + k); parts.append(Jc); out['sample'][c] = inf
        assert np.all(Jc['label'] == lab)
    from interp.part_core import cat
    J = cat(parts); del parts; y = J['label'].astype(int)
    P0, M0 = H.p4(J); Jid = H.rebuild(J, P0, M0); O0 = H.obs(Jid); del P0, M0
    log(f'sample {len(y)} jets')
    rng = np.random.default_rng(4870)
    # ---- build windows
    ARM = {}
    hidx_all = hadron_index(J)
    for c in SIGC:
        X = EDGE[c]
        wins = [(f'L{L:+d}', X + L, X + L + 10, 20.0) for L in LS] + [('reverse', X + 7, X + 27, -20.0)]
        for wn, lo_, hi_, dm in wins:
            sel = np.flatnonzero((y == SIGC[c]) & (O0['has'] > 0) & (hidx_all >= 0) & (O0['m'] >= lo_) & (O0['m'] < hi_))
            if len(sel) > CAP: sel = np.sort(rng.choice(sel, CAP, replace=False))
            if len(sel) < 5: continue
            Ja = H._sub(J, sel)
            for ed in EDITS:
                pi = H.lepton_index(Ja) if ed == 'lep' else hadron_index(Ja)
                V, m, ok, bk = edit_one(Ja, pi, dm, rng)
                if ok.sum() < 5: continue
                Jok = H._sub(Ja, ok)
                Jv = {v: H.rebuild(Jok, V[v][ok], m[ok]) for v in ('E', 'C')}
                Ov = {v: H.obs(Jv[v]) for v in ('E', 'C')}
                rows = sel[ok]
                info = dict(n_window=int(len(sel)), n=int(ok.sum()), dm=dm, window=[lo_, hi_],
                            mean_theta=float(bk['theta'][ok].mean()), mean_theta_E=float(bk['theta_E'][ok].mean()),
                            E_dm=float((Ov['E']['m'] - O0['m'][rows]).mean()), C_dm=float((Ov['C']['m'] - O0['m'][rows]).mean()),
                            E_diso=float((Ov['E']['iso'] - O0['iso'][rows]).mean()), C_diso=float((Ov['C']['iso'] - O0['iso'][rows]).mean()),
                            E_df=float((Ov['E']['f'] - O0['f'][rows]).mean()), C_df=float((Ov['C']['f'] - O0['f'][rows]).mean()),
                            lepton_still_hardest_lepton=float(np.mean(H.lepton_index(Jv['E']) >= 0)))
                ARM[(c, wn, ed)] = dict(rows=rows, J=Jv, O=Ov, info=info)
                log(f'arm {c} {wn} {ed}: ' + json.dumps(info))
    out['arms_info'] = {'|'.join(k): A['info'] for k, A in ARM.items()}
    # ---- networks
    SC = {}; out['unedited'] = {}
    for name in NETS:
        Mdl = H.load_net(name); Lid = H.logits(Mdl, Jid); SC[(name, 'id')] = Lid
        out['unedited'][name] = dict(acc_10class=float((Lid.argmax(1) == y).mean()),
                                     acc_by_class={c: float((Lid.argmax(1) == lab)[y == lab].mean()) for c, _, lab in CLS})
        log(f'{name}: ' + json.dumps(out['unedited'][name]))
        for k, A in ARM.items():
            for v in ('E', 'C'): SC[(name,) + k + (v,)] = H.logits(Mdl, A['J'][v])
        del Mdl
    REF = fit_refs(out)
    RL = {}
    for c in SIGC:
        for r in REFS: RL[(r, c, 'id')] = H.ref_lo(REF, c, r, O0)
    for k, A in ARM.items():
        for r in REFS:
            for v in ('E', 'C'): RL[(r,) + k + (v,)] = H.ref_lo(REF, k[0], r, A['O'][v])

    def ec(mname, k):
        c = k[0]; rows = ARM[k]['rows']
        if mname in REFS: return (RL[(mname,) + k + ('E',)] - RL[(mname,) + k + ('C',)])
        return H.score(SC[(mname,) + k + ('E',)], c) - H.score(SC[(mname,) + k + ('C',)], c)

    rs = np.random.default_rng(4899); res = {}; fam = []
    EC = {}
    for k in ARM:
        for mname in NETS + REFS:
            x = ec(mname, k); EC[(mname,) + k] = x; mu, ci = boot(x, rs)
            res['|'.join(k) + f'|E-C|{mname}'] = dict(mean=mu, ci=ci, n=len(x))
        for mname in TESTED:
            d = EC[(mname,) + k] - EC[('MLP_m',) + k]; mu, ci = boot(d, rs)
            res['|'.join(k) + f'|D_MLPm|{mname}'] = dict(mean=mu, ci=ci, n=len(d), p_perm=signflip(d, rs))
            d2 = EC[(mname,) + k] - EC[('MLP_mlep',) + k]; mu, ci = boot(d2, rs)
            res['|'.join(k) + f'|D_MLPmlep|{mname}'] = dict(mean=mu, ci=ci, n=len(d2))
    # ---- H48 arms reproduced and edge-localization contrasts
    def pool(mname, c, ed, wl, ref='MLP_m'):
        xs = [EC[(mname, c, w, ed)] - (EC[(ref, c, w, ed)] if ref else 0) for w in wl if (mname, c, w, ed) in EC]
        return np.concatenate(xs) if xs else np.array([])
    BELOW, CROSS, ABOVE = ('L-43', 'L-33'), ('L-13',), ('L+7', 'L+17')
    for c in SIGC:
        for ed in EDITS:
            for mname in NETS + REFS:
                a, b_ = pool(mname, c, ed, CROSS, None), pool(mname, c, ed, BELOW, None)
                res[f'{c}|{ed}|across-below|E-C|{mname}'] = contrast(a, b_, rs)
            for mname in TESTED:
                for nm, other in (('cross-below', BELOW), ('cross-above', ABOVE)):
                    for ref in ('MLP_m', 'MLP_mlep'):
                        key = f'{c}|{ed}|{nm}|D_{ref}|{mname}'
                        res[key] = contrast(pool(mname, c, ed, CROSS, ref), pool(mname, c, ed, other, ref), rs)
                        if ref == 'MLP_m' and min(res[key]['n']) >= 20: fam.append(key)
                    for ref in ('MLP_m',):
                        pass
                # above vs below (is the excess present above the edge as well)
                key = f'{c}|{ed}|above|D_MLP_m|{mname}'; x = pool(mname, c, ed, ABOVE, 'MLP_m'); mu, ci = boot(x, rs)
                res[key] = dict(mean=mu, ci=ci, n=len(x), p_perm=signflip(x, rs) if len(x) >= 5 else 1.0)
                key = f'{c}|{ed}|cross|D_MLP_m|{mname}'; x = pool(mname, c, ed, CROSS, 'MLP_m'); mu, ci = boot(x, rs)
                res[key] = dict(mean=mu, ci=ci, n=len(x), p_perm=signflip(x, rs) if len(x) >= 5 else 1.0); fam.append(key)
                key = f'{c}|{ed}|below|D_MLP_m|{mname}'; x = pool(mname, c, ed, BELOW, 'MLP_m'); mu, ci = boot(x, rs)
                res[key] = dict(mean=mu, ci=ci, n=len(x), p_perm=signflip(x, rs) if len(x) >= 5 else 1.0)
    adj = H.holm([res[k]['p_perm'] for k in fam])
    for k, a_ in zip(fam, adj): res[k]['p_holm'] = float(a_)
    out['holm_family'] = fam; out['results'] = res
    # log summary
    for c in SIGC:
        for ed in EDITS:
            row = []
            for L in LS:
                k = (c, f'L{L:+d}', ed)
                if k in ARM: row.append(f'L{L:+d}(n{ARM[k]["info"]["n"]}) ' + ' '.join(f'{res["|".join(k) + "|E-C|" + n]["mean"]:+.2f}' for n in ('ParT_full', 'PN_full', 'ParT_kin', 'PN_kin', 'MLP_m', 'MLP_mlep')))
            log(f'{c} {ed} scan [E-C ParT_full PN_full ParT_kin PN_kin MLP_m MLP_mlep]:\n   ' + '\n   '.join(row))
            log(f'{c} {ed} across-below E-C: ' + ' '.join(f'{n} {res[f"{c}|{ed}|across-below|E-C|{n}"]["mean"]:+.2f}' for n in NETS + REFS))
            log(f'{c} {ed} D_MLPm below/cross/above: ' + ' | '.join(f'{n} {res[f"{c}|{ed}|below|D_MLP_m|{n}"]["mean"]:+.2f}/{res[f"{c}|{ed}|cross|D_MLP_m|{n}"]["mean"]:+.2f}/{res[f"{c}|{ed}|above|D_MLP_m|{n}"]["mean"]:+.2f}' for n in TESTED))
    (RES / f'{TAG}.json').write_text(json.dumps(H.clean(out), indent=1))
    keep = dict(y=y, m0=O0['m'])
    for k, x in EC.items(): keep['EC|' + '|'.join(k)] = np.asarray(x, np.float32)
    for k, A in ARM.items(): keep['rows|' + '|'.join(k)] = A['rows']
    np.savez_compressed(RES / f'{TAG}_perjet.npz', **keep); log('saved')
    try:
        fig(res, ARM)
    except Exception as e: log('figure failed', repr(e))


def fig(res, ARM):
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5), sharex='col')
    cols = dict(ParT_full='#2f5d8a', PN_full='#c0603a', ParT_kin='#7fa7cc', PN_kin='#e0a080', MLP_m='#555555', MLP_mlep='#5a9a5a')
    for j, c in enumerate(('Hqql', 'Tbl')):
        X = EDGE[c]
        for i, ed in enumerate(EDITS):
            ax = axs[i, j]
            for n, col in cols.items():
                xs, ys, lo_, hi_ = [], [], [], []
                for L in LS:
                    k = f'{c}|L{L:+d}|{ed}|E-C|{n}'
                    if k in res and res[k]['n'] >= 10:
                        xs.append(X + L + 5); ys.append(res[k]['mean']); lo_.append(res[k]['ci'][0]); hi_.append(res[k]['ci'][1])
                if xs:
                    ax.plot(xs, ys, '-o', ms=3, color=col, label=n, ls='--' if n.startswith('MLP') else '-')
                    ax.fill_between(xs, lo_, hi_, color=col, alpha=0.15, lw=0)
            ax.axvspan(X - 20, X, color='k', alpha=0.07); ax.axhline(0, color='k', lw=0.5)
            ax.set_title(f'{"H→ℓνqq′, edge 125 GeV" if c == "Hqql" else "t→bℓν, edge 152.6 GeV"}; '
                         f'{"lepton rotated" if ed == "lep" else "hardest hadron rotated"} (+20 GeV)', fontsize=10)
            if j == 0: ax.set_ylabel('E − C: change in signal log-odds')
            if i == 1: ax.set_xlabel('unedited jet mass, window centre [GeV] (shaded: windows whose +20 GeV shift crosses the edge)')
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(RES / f'{TAG}_scan.png', dpi=120); plt.close(fig); log('figure saved')


if __name__ == '__main__':
    main()
