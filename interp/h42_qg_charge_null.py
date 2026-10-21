"""H42 (loop D, claim D16): nulls for the jet-charge response of the QG ParT / ParticleNet kinpid taggers.

D16 (H40 / H40v): charge conjugation of every charged particle (E1, four-momenta fixed) moves the quark score of QG ParT kinpid
and ParticleNet kinpid by about -0.24 log-odds per sign of the jet charge Q_0.3. H42 asks whether that response (i) is a
property of the training sample (quark jets in the Komiske-Metodiev-Thaler Pythia Z+jet sample carry net positive charge),
(ii) reproduces with a second training seed and (iii) is absent when the labels are independent of the class.

Q_kappa = sum_i q_i (pT_i / pT_J)^kappa, kappa = 0.3 (also 0.5); pT_J = |vector sum of the particle pT|; q_i from the PDG id.
Score s = log p(quark) - log p(gluon) (positive = more quark-like).

Samples
  py   interp.data_qg.sample(per_class=10000, seed=4201): KMT Pythia 8 test split, 10,000 quark + 10,000 gluon random rows
  hw   Herwig 7.1 test jets (Zenodo 3066475 files 18-19), 10,000 per class, seed 4202, via h14.load_sample('hw')
  full splits for (a): all 199,998 Pythia test jets; all 200,000 Herwig jets of files 18-19
Models (all trained on Pythia; weaver checkpoints in /j-jepa-vol/aaron-interp/models/qg/<name>)
  part_kinpid, pnet_kinpid             H2 trainings (seed not set = "seed 0" here)
  part_kinpid_s1, pnet_kinpid_s1       H42 trainings, torch seed 1, authors' settings (interp/h42_kinpid_train.sh)
  part_kinpid_shuf, pnet_kinpid_shuf   H42 trainings, label = hash(jet_pt, jet_eta, jet_phi) mod 2 (independent of the class)
References
  LR_Q03_py  logistic regression on Q_0.3, fitted on Pythia test-split jets not in the py sample (all of them)
  LR_Q03_hw  logistic regression on Q_0.3, fitted on Herwig file 0 (100,000 jets; files 18-19 are the evaluation files)
Edits (interp.h40_qg_pid_charge.make_edits; PDG ids only, (pT, y, phi) of every particle and N fixed)
  E1   every charged particle pid -> -pid (photons, K0L untouched): Q_kappa -> -Q_kappa exactly (the charge-conjugate jet)
  E2k1 the hardest charged particle pid -> -pid (Q moves by -2 q_1 z_1^kappa);  C2: control, one + and one - particle flipped
       together with |Delta Q_0.3|, |Delta Q_0.5| < 0.01 (same kind of relabel, jet charge fixed)
  E3g_4 / E3n_4  2 pi+ and 2 pi- below the jet's median particle pT -> photon / K0L;  C3_4: the same 4 pions pi+ <-> pi-
Analyses
  (a) sample asymmetry: mean Q_0.3 (and Q_0.5) per class and generator, bootstrap 95 % CI, fraction Q > 0, Welch t of
      quark minus gluon mean; E1 maps the quark distribution onto its mirror image.
  (b) C2ST (classifier two-sample test): per class and generator, the class's eligible jets are split at random into two
      halves; half B is edited, half A is kept. A GBDT on jet-level features (Q_0.3, Q_0.5, n+, n-, n_ch, f_ch, species
      counts, N, girth, C1^{0.5,1,2}, pT^D, mass) is trained on 50 % of A u B to tell edited from unedited and scored on the
      other 50 %: AUC (0.5 = indistinguishable), bootstrap CI over the held-out jets, permutation null (edited / unedited
      labels permuted, GBDT retrained, P = 50). Edits E1 (both generators), E3g_4, E3n_4 (Pythia).
      Analytic reference for E1: AUC of Q vs -Q within the class, = the charge asymmetry of that class as an AUC.
  (c) models on both generators: E1 x sign(Q_0.3), E1 split by class x sign Q, E2k1 - C2 x sign(charge of the hardest charged
      particle), slopes per unit Delta Q_0.3, E3 - C3 (Pythia); McNemar at the threshold giving 50 % quark efficiency on
      unedited jets; 95 % bootstrap CI (B = 1000), sign-flip permutation p and normal-approximation p, Holm over all
      (model x generator x contrast) cells.
  (d) decision for the retrained models (when final): H0 (seed artefact) if seed-1 E1 x sign Q is outside [0.5, 1.5] x
      seed-0 or of opposite sign; shuffled-label models compared with the H40v band of 12 permuted networks.
Steps
  --step run  [--models a,b,...]  (CPU, conda + env-part) -> $AI_RESULTS/physics/h42_qg_charge_null.json, h42_perjet_<gen>.npz
               models not final are skipped and listed
  --step run --gens py --tag _py / --gens hw --tag _hw (two jobs), then --step merge (no --tag): Holm over both generators
  --step fig  (CPU, jax env) results/physics/h42_charge_asym.png
  --smoke N   N jets per class, suffix _smoke, C2ST P = 5
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import h40_qg_pid_charge as H40  # noqa: E402

T0 = time.time()
log = lambda *a: print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)
V = Path('/j-jepa-vol/aaron-interp')
RES = Path(os.environ.get('AI_RESULTS', V / 'results')) / 'physics'
SEED_PY, SEED_HW = 4201, 4202
ARCH = dict(part='part', pnet='pn')
ALL_MODELS = ['part_kinpid', 'pnet_kinpid', 'part_kinpid_s1', 'pnet_kinpid_s1', 'part_kinpid_shuf', 'pnet_kinpid_shuf']
EDITS = dict(py=['E1', 'E2k1', 'C2', 'E3g_4', 'E3n_4', 'C3_4'], hw=['E1', 'E2k1', 'C2'])
KNOWN = dict(py=dict(part_kinpid='AUC 0.9104 / acc on H40 sample (seed 4001); 0.9123 full test', pnet_kinpid='AUC 0.9058 (H40 sample)'),
             hw=dict(part_kinpid='AUC 0.8029 (H40 Herwig sample)', pnet_kinpid='AUC 0.8047 (H40 Herwig sample)'))
PERM_BAND = dict(part=0.0194, pnet=0.1123)   # H40v: max |E1 x sign Q| over 6 weight-permuted copies each (log-odds)


def qcharge(pid):
    return H40.charge(pid)


def q_from_arrays(px, py, pid, kappa):
    """(J, L) px, py, pid -> Q_kappa per jet"""
    pt = np.hypot(px, py); ptj = np.hypot(px.sum(1), py.sum(1))
    z = np.where(pt > 0, pt / np.maximum(ptj, 1e-12)[:, None], 0.0)
    return (qcharge(pid) * z ** kappa).sum(1)


# ===================================================================================================== (a) full-split charge
def full_split_q():
    from interp import data_qg
    from interp import h14_qg_generator as H14
    out = {}
    d = data_qg.DST / 'test'; y = np.load(d / 'label.npy').astype(int)
    F = np.load(d / 'full.npy', mmap_mode='r'); P = np.load(d / 'pid.npy', mmap_mode='r')
    Q3, Q5 = [], []
    for s in range(0, len(y), 20000):
        f = np.asarray(F[s:s + 20000], np.float64); p = np.asarray(P[s:s + 20000]).astype(np.int64)
        Q3.append(q_from_arrays(f[..., 0], f[..., 1], p, 0.3)); Q5.append(q_from_arrays(f[..., 0], f[..., 1], p, 0.5))
    out['py'] = dict(y=y, Q03=np.concatenate(Q3), Q05=np.concatenate(Q5), source='KMT Pythia test split (all rows)')
    log(f'(a) Pythia full test split: {len(y)} jets')
    ys, q3, q5 = [], [], []
    for i in (18, 19):
        with np.load(H14._hw_file(i)) as z:
            X = z['X'].astype(np.float64); yy = z['y'].astype(int)
        px, py_ = X[..., 0] * np.cos(X[..., 2]), X[..., 0] * np.sin(X[..., 2]); pid = X[..., 3].astype(np.int64) * (X[..., 0] > 0)
        q3.append(q_from_arrays(px, py_, pid, 0.3)); q5.append(q_from_arrays(px, py_, pid, 0.5)); ys.append(yy)
    out['hw'] = dict(y=np.concatenate(ys), Q03=np.concatenate(q3), Q05=np.concatenate(q5), source='Herwig 7.1 files 18-19 (all rows)')
    log(f'(a) Herwig files 18-19: {len(out["hw"]["y"])} jets')
    return out


def asym_stats(y, Q3, Q5, B=1000, seed=SEED_PY):
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(seed); D = {}
    for c, cn in ((1, 'quark'), (0, 'gluon')):
        d = {'n': int((y == c).sum())}
        for k, v in (('Q03', Q3[y == c]), ('Q05', Q5[y == c])):
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
            d[f'mean_{k}'] = float(v.mean()); d[f'ci_{k}'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            d[f'sd_{k}'] = float(v.std()); d[f'frac_{k}_pos'] = float((v > 0).mean()); d[f'frac_{k}_neg'] = float((v < 0).mean())
            d[f'frac_{k}_zero'] = float((v == 0).mean())
            # sign test of mean 0 (the E1 mirror image has mean -mean): t statistic
            d[f't_mean0_{k}'] = float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
        D[cn] = d
    for k, (a, b) in (('Q03', (Q3[y == 1], Q3[y == 0])), ('Q05', (Q5[y == 1], Q5[y == 0]))):
        t = ttest_ind(a, b, equal_var=False); D[f'diff_{k}'] = dict(mean=float(a.mean() - b.mean()), t=float(t.statistic), p=float(t.pvalue))
    return D


# ================================================================================================================== (b) C2ST
C2ST_FEATS = ['Q03', 'Q05', 'n_plus', 'n_minus', 'n_ch', 'fch', 'n_pi', 'n_K', 'n_p', 'n_gam', 'n_K0L', 'n_n', 'n_e', 'n_mu',
              'N', 'girth', 'C1_05', 'C1_1', 'C1_2', 'ptd', 'mass']


def c2st(F0, Fe, mask, seed, P=50, B=1000):
    """F0 / Fe: feature dicts of unedited / edited jets (same jets); mask: jets eligible (edit applied). Jets split in halves:
    half A contributes its unedited version, half B its edited version (each jet appears once)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); ix = rng.permutation(np.flatnonzero(mask)); h = len(ix) // 2
    A, Bx = ix[:h], ix[h:2 * h]
    X = np.concatenate([np.stack([F0[k][A] for k in C2ST_FEATS], 1), np.stack([Fe[k][Bx] for k in C2ST_FEATS], 1)]).astype(np.float64)
    lab = np.r_[np.zeros(h, int), np.ones(h, int)]
    sp = rng.permutation(2 * h); tr, te = sp[:h], sp[h:]

    def fit_auc(lb):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X[tr], lb[tr])
        return m.predict_proba(X[te])[:, 1]

    p1 = fit_auc(lab); auc = float(roc_auc_score(lab[te], p1))
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(te), len(te))
        if lab[te][k].min() == lab[te][k].max(): continue
        bs.append(roc_auc_score(lab[te][k], p1[k]))
    null = []
    for _ in range(P):
        lp = rng.permutation(lab); null.append(roc_auc_score(lp[te], fit_auc(lp)))
    null = np.array(null)
    return dict(n_jets_per_half=int(h), n_test=int(len(te)), auc=auc, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                null_mean=float(null.mean()), null_sd=float(null.std()), null_max=float(null.max()), null_q95=float(np.percentile(null, 95)),
                p_perm=float((1 + (null >= auc).sum()) / (P + 1)),
                z_vs_null=float((auc - null.mean()) / max(null.std(), 1e-9)), P=P)


def auc_mirror(Q):
    """AUC for separating -Q (edited) from Q (unedited) by the value of Q alone, folded to >= 0.5 (= 0.5 when symmetric)"""
    from sklearn.metrics import roc_auc_score
    if not len(Q): return 0.5
    a = float(roc_auc_score(np.r_[np.zeros(len(Q)), np.ones(len(Q))], np.r_[Q, -Q])); return max(a, 1 - a)


# ============================================================================================================ (c) models
def load_models(names, dev):
    from interp import h5_qg_counting as H
    M = H.Models(dev); used, skipped = [], {}
    for nm in names:
        d = V / 'models' / 'qg' / nm
        if not (d / 'net_best_epoch_state.pt').exists(): skipped[nm] = 'no checkpoint'; continue
        n_ep = len(list(d.glob('net_epoch-*_state.pt')))
        if not ((d / 'predict_output').exists() or n_ep >= 20): skipped[nm] = f'not final ({n_ep} epochs written)'; continue
        M.add_weaver(nm, ARCH[nm.split('_')[0]], True); used.append(nm)
    log('models used', used, 'skipped', skipped)
    return M, used, skipped


def fit_lr(Q, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=2000, C=1e3).fit(Q[:, None], y)
    return dict(coef=float(m.coef_[0, 0]), intercept=float(m.intercept_[0]), n=int(len(y)))


def lr_score(lr, Q):
    return lr['coef'] * Q + lr['intercept']


def run_gen(gen, M, models, LR, per, B, P):
    from interp import h5_qg_counting as H
    from interp import h14_qg_generator as H14
    L0, y, info = H14.load_sample(gen, per, SEED_PY if gen == 'py' else SEED_HW)
    y = np.asarray(y).astype(int); J = len(y)
    log(f'{gen}: {J} jets ({info["source"]}), quark fraction {y.mean():.3f}')
    E = H40.make_edits(L0, gen + '_h42', full=False); E = {k: E[k] for k in EDITS[gen]}
    log(f'{gen}: edits applied ' + ', '.join(f'{k} {int(v[1].sum())}' for k, v in E.items()))
    out = dict(y=y, idx=np.asarray(info['idx']))
    J0 = H.from_list(L0, Lpad=180); F0, shp = H40.all_feats(J0)
    for k, v in F0.items(): out[f'F|base|{k}'] = np.asarray(v, np.float64)
    out['F|base|qlead'] = np.array([(lambda q: q[q != 0][0] if (q != 0).any() else 0.0)(qcharge(a[:, 3])) for a in L0])
    t = time.time()
    for nm in models: out[f'base|{nm}'] = -M.logit(nm, J0); log(f'{gen} base {nm} scored ({time.time() - t:.0f}s)')
    for ln, lr in LR.items(): out[f'base|{ln}'] = lr_score(lr, out['F|base|Q03'])
    refs = list(LR)
    unedited = {nm: H40.roc(out[f'base|{nm}'], y) for nm in models + refs}
    for nm in models + refs: log(f'{gen} unedited {nm}: ' + ', '.join(f'{k} {v:.4f}' for k, v in unedited[nm].items()))
    Fe_all = {}
    for en, (Le, ok, nr) in E.items():
        Je = H.from_list(Le, Lpad=180); Fe, _ = H40.all_feats(Je, shp); Fe_all[en] = Fe
        assert np.allclose(Je['pt'], J0['pt']) and np.allclose(Je['eta'], J0['eta']) and np.allclose(Je['phi'], J0['phi'])
        out[f'ok|{en}'] = ok
        for k in ('Q03', 'Q05', 'fch', 'n_ch'): out[f'F|{en}|{k}'] = np.asarray(Fe[k], np.float64)
        for nm in models: out[f'{en}|{nm}'] = -M.logit(nm, Je)
        for ln, lr in LR.items(): out[f'{en}|{ln}'] = lr_score(lr, out[f'F|{en}|Q03'])
        log(f'{gen} {en}: applied {int(ok.sum())}; mean dQ03 {np.mean((out[f"F|{en}|Q03"] - out["F|base|Q03"])[ok]):+.4f}; '
            + '; '.join(f'{nm} {np.mean((out[f"{en}|{nm}"] - out[f"base|{nm}"])[ok]):+.4f}' for nm in models + refs))
    if 'E1' in E:
        mx = np.abs(out['F|E1|Q03'] + out['F|base|Q03']).max(); log(f'{gen} check E1: max |Q_E1 + Q| = {mx:.2e}')
    # (b) C2ST
    C = {}
    for en in [e for e in ('E1', 'E3g_4', 'E3n_4') if e in E]:
        for c, cn in ((1, 'quark'), (0, 'gluon')):
            mk = E[en][1] & (y == c)
            r = c2st(F0, Fe_all[en], mk, seed=(SEED_PY if gen == 'py' else SEED_HW) + 10 * c + len(en), P=P, B=B)
            if en == 'E1': r['analytic_auc_Q_vs_minusQ'] = auc_mirror(out['F|base|Q03'][mk])
            C[f'{en}|{cn}'] = r
            log(f'{gen} C2ST {en} {cn}: AUC {r["auc"]:.4f} {np.round(r["ci"], 4).tolist()} null {r["null_mean"]:.4f}+-{r["null_sd"]:.4f} '
                f'(max {r["null_max"]:.4f}) p {r["p_perm"]:.3f}' + (f'; analytic Q vs -Q {r["analytic_auc_Q_vs_minusQ"]:.4f}' if en == 'E1' else ''))
    S = model_stats(out, y, unedited, models + refs, gen, B)
    return out, dict(unedited=unedited, known=KNOWN.get(gen, {}), c2st=C, stats=S,
                     sample=dict(J=J, per_class=per, seed=SEED_PY if gen == 'py' else SEED_HW, source=info['source']),
                     applied={k: int(v[1].sum()) for k, v in E.items()},
                     asym_sample=asym_stats(y, out['F|base|Q03'], out['F|base|Q05'], B=B))


def model_stats(out, y, unedited, names, gen, B):
    from scipy.stats import norm
    bt = H40.Boot(B=B); S = {}
    sq = np.sign(out['F|base|Q03']); sl = np.sign(out['F|base|qlead'])
    defs = [('E1_signQ', 'E1', None, sq), ('E2k1-C2_signlead', 'E2k1', 'C2', sl)]
    if 'ok|E3g_4' in out: defs += [('E3g_4-C3', 'E3g_4', 'C3_4', None), ('E3n_4-C3', 'E3n_4', 'C3_4', None)]
    for cn, en, ctl, sg in defs:
        ok = out[f'ok|{en}'] & (out[f'ok|{ctl}'] if ctl else True) & ((sg != 0) if sg is not None else True)
        row = dict(n_jets=int(ok.sum()))
        for nm in names:
            base = out[f'base|{nm}']; de = out[f'{en}|{nm}'] - base; dc = (out[f'{ctl}|{nm}'] - base) if ctl else np.zeros_like(de)
            d = (de - dc)[ok] * (sg[ok] if sg is not None else 1)
            se = d.std(ddof=1) / np.sqrt(len(d)); z = d.mean() / se if se > 0 else 0.0
            r = dict(mean=float(d.mean()), ci=bt.ci(d), p_perm=bt.p(d), p_t=float(2 * norm.sf(abs(z))) if se > 0 else 1.0,
                     mean_over_sd=float(d.mean() / max(np.std(base), 1e-12)))
            thr = unedited[nm]['thr50']; a = (base + dc >= thr)[ok]; b = (base + de >= thr)[ok]
            n01, n10, pm = H40.mcnemar(a, b); r['mcnemar'] = dict(to_quark=n01, to_gluon=n10, p=pm, calls_changed_pct=float(100 * (a != b).mean()))
            if sg is not None:
                r['split'] = {}
                for c, lab in ((1, 'q'), (0, 'g')):
                    for s_, sn in ((1, 'pos'), (-1, 'neg')):
                        mk = ok & (y == c) & (sg == s_)
                        if mk.sum() > 10: r['split'][f'{lab}_{sn}'] = dict(n=int(mk.sum()), mean=float(de[mk].mean()), ci=bt.ci(de[mk]))
            row[nm] = r
        S[cn] = row
    # slopes per unit Delta Q_0.3
    S['slope'] = {}
    for nm in names:
        u = {}
        for en in ('E1', 'E2k1'):
            ok = out[f'ok|{en}']
            u[f'{en}_per_dQ03'] = H40.slope_ci((out[f'F|{en}|Q03'] - out['F|base|Q03'])[ok], (out[f'{en}|{nm}'] - out[f'base|{nm}'])[ok], B=300)
        S['slope'][nm] = u
    return S


def holm_all(res, models):
    pl = []
    for gen in ('py', 'hw'):
        if gen not in res: continue
        for cn, row in res[gen]['stats'].items():
            if cn == 'slope': continue
            for nm in models:
                if nm in row: pl.append((gen, cn, nm, row[nm]['p_t']))
    adj = H40.holm([p for *_, p in pl])
    for (gen, cn, nm, _), a in zip(pl, adj): res[gen]['stats'][cn][nm]['p_t_holm'] = float(a)
    # C2ST cells
    pc = [(gen, k, v['p_perm']) for gen in ('py', 'hw') if gen in res for k, v in res[gen]['c2st'].items()]
    adjc = H40.holm([p for *_, p in pc])
    for (gen, k, _), a in zip(pc, adjc): res[gen]['c2st'][k]['p_perm_holm'] = float(a)
    # the permutation p has a floor 1 / (P + 1); also a one-sided normal p from z = (AUC - null mean) / null sd, Holm over cells
    from scipy.stats import norm
    pz = [(gen, k, float(norm.sf(v['z_vs_null']))) for gen in ('py', 'hw') if gen in res for k, v in res[gen]['c2st'].items()]
    adjz = H40.holm([p for *_, p in pz])
    for (gen, k, p), a in zip(pz, adjz): res[gen]['c2st'][k]['p_z'] = p; res[gen]['c2st'][k]['p_z_holm'] = float(a)
    return dict(model_cells=len(pl), model_reject_05=int((adj < 0.05).sum()), c2st_cells=len(pc), c2st_reject_05=int((adjc < 0.05).sum()),
                c2st_reject_05_z=int((adjz < 0.05).sum()))


def decisions(res, used):
    D = {}
    for gen in ('py', 'hw'):
        if gen not in res: continue
        st = res[gen]['stats']['E1_signQ']; d = {}
        for arch in ('part', 'pnet'):
            s0 = st.get(f'{arch}_kinpid', {}).get('mean')
            for var in ('s1', 'shuf'):
                nm = f'{arch}_kinpid_{var}'
                if nm not in st or s0 is None: continue
                v = st[nm]['mean']; r = v / s0
                if var == 's1':
                    d[nm] = dict(e1_signQ=v, seed0=s0, ratio=r, within_0p5_1p5_same_sign=bool(0.5 <= r <= 1.5),
                                 outcome='H1 (same sign, within [0.5, 1.5] x seed 0)' if 0.5 <= r <= 1.5 else 'H0 (outside [0.5, 1.5] x seed 0 or opposite sign)')
                else:
                    d[nm] = dict(e1_signQ=v, ci=st[nm]['ci'], perm_band_abs=PERM_BAND[arch], within_perm_band=bool(abs(v) <= PERM_BAND[arch]),
                                 auc=res[gen]['unedited'][nm]['AUC'])
        if 'LR_Q03_py' in st:
            for nm in [m for m in used if m in st]: d.setdefault(nm, {})['ratio_to_LR_Q03_py'] = st[nm]['mean'] / st['LR_Q03_py']['mean']
            for nm in [m for m in used if m in st]: d[nm]['ratio_to_LR_Q03_hw'] = st[nm]['mean'] / st['LR_Q03_hw']['mean']
        D[gen] = d
    return D


def finish(res, used, LR):
    res['holm'] = holm_all(res, used); log('Holm', res['holm'])
    res['decisions'] = decisions(res, used); log('decisions', json.dumps(res['decisions'], default=float))


def step_merge(a):
    """combine the per-generator runs (--tag <t>_py, <t>_hw) into h42_qg_charge_null<t>.json, add Holm and decisions"""
    sm = '_smoke' if a.smoke else ''
    Rp = json.loads((RES / f'h42_qg_charge_null{a.tag}_py{sm}.json').read_text()); Rh = json.loads((RES / f'h42_qg_charge_null{a.tag}_hw{sm}.json').read_text())
    res = dict(Rp); res['hw'] = Rh['hw']; assert Rp['used'] == Rh['used']
    finish(res, res['used'], res['LR'])
    for gen in ('py', 'hw'):
        for nm in res['used'] + list(res['LR']):
            r = res[gen]['stats']['E1_signQ'][nm]
            log(f'{gen} E1 x signQ {nm}: {r["mean"]:+.4f} {np.round(r["ci"], 4).tolist()} p_holm {r.get("p_t_holm", np.nan):.2e}')
    (RES / f'h42_qg_charge_null{a.tag}{sm}.json').write_text(json.dumps(res, indent=1, default=float)); log('wrote', RES / f'h42_qg_charge_null{a.tag}{sm}.json')


def step_run(a):
    import torch
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; RES.mkdir(parents=True, exist_ok=True)
    nth = int(os.environ.get('OMP_NUM_THREADS', '4')); torch.set_num_threads(nth)
    sm = '_smoke' if a.smoke else ''; per = a.smoke or 10000; B = 200 if a.smoke else 1000; P = 5 if a.smoke else 50
    tag = a.tag
    log(f'device {dev}, threads {nth}; per class {per}; B {B}; C2ST permutations {P}; tag {tag!r}')
    names = a.models.split(',') if a.models else ALL_MODELS
    M, used, skipped = load_models(names, dev)
    res = dict(models=M.info, used=used, skipped=skipped, seeds=dict(py=SEED_PY, hw=SEED_HW))
    # (a) full splits
    FQ = full_split_q(); res['asym_full'] = {g: dict(source=v['source'], **asym_stats(v['y'], v['Q03'], v['Q05'])) for g, v in FQ.items()}
    for g, v in res['asym_full'].items():
        log(f'(a) {g}: <Q03> quark {v["quark"]["mean_Q03"]:+.4f} {np.round(v["quark"]["ci_Q03"], 4).tolist()}, gluon {v["gluon"]["mean_Q03"]:+.4f} '
            f'{np.round(v["gluon"]["ci_Q03"], 4).tolist()}; frac>0 q {v["quark"]["frac_Q03_pos"]:.4f} g {v["gluon"]["frac_Q03_pos"]:.4f}; diff t {v["diff_Q03"]["t"]:.1f}')
    # histograms for the figure
    bins = np.linspace(-2.5, 2.5, 101); H = {}
    for g, v in FQ.items():
        for c, cn in ((1, 'quark'), (0, 'gluon')):
            H[f'{g}|{cn}'] = np.histogram(v['Q03'][v['y'] == c], bins)[0].tolist()
    res['hist_Q03'] = dict(bins=bins.tolist(), counts=H)
    # LR references: Pythia = test split rows outside the py sample; Herwig = file 0
    from interp import data_qg
    from interp import h14_qg_generator as H14
    S = data_qg.sample(per, SEED_PY); free = np.ones(len(FQ['py']['y']), bool); free[S['idx']] = False
    LR = dict(LR_Q03_py=fit_lr(FQ['py']['Q03'][free], FQ['py']['y'][free]))
    with np.load(H14._hw_file(0)) as z:
        X = z['X'].astype(np.float64); yy = z['y'].astype(int)
    q0 = q_from_arrays(X[..., 0] * np.cos(X[..., 2]), X[..., 0] * np.sin(X[..., 2]), X[..., 3].astype(np.int64) * (X[..., 0] > 0), 0.3)
    LR['LR_Q03_hw'] = fit_lr(q0, yy); del X
    res['LR'] = LR; log('LR references', LR)
    gens = a.gens.split(',')
    for gen in gens:
        out, st = run_gen(gen, M, used, LR, per, B, P); res[gen] = st
        np.savez_compressed(RES / f'h42_perjet_{gen}{tag}{sm}.npz', **{k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in out.items()})
        (RES / f'h42_qg_charge_null{tag}{sm}.json').write_text(json.dumps(res, indent=1, default=float))
    if len(gens) == 2: finish(res, used, LR)
    for gen in gens:
        for nm in used + list(LR):
            r = res[gen]['stats']['E1_signQ'][nm]
            log(f'{gen} E1 x signQ {nm}: {r["mean"]:+.4f} {np.round(r["ci"], 4).tolist()} p_t {r["p_t"]:.2e}; '
                f'McNemar {r["mcnemar"]["calls_changed_pct"]:.2f} % changed; split ' + ', '.join(f'{k} {v["mean"]:+.3f}' for k, v in r['split'].items()))
    (RES / f'h42_qg_charge_null{tag}{sm}.json').write_text(json.dumps(res, indent=1, default=float))
    log('wrote', RES / f'h42_qg_charge_null{tag}{sm}.json')


# ================================================================================================================== figure
def step_fig(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sm = '_smoke' if a.smoke else ''
    R = json.loads((RES / f'h42_qg_charge_null{a.tag}{sm}.json').read_text())
    b = np.array(R['hist_Q03']['bins']); c = 0.5 * (b[1:] + b[:-1]); w = b[1] - b[0]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, g, gl in zip(axs, ('py', 'hw'), ('Pythia 8 (training generator)', 'Herwig 7.1')):
        A = R['asym_full'][g]
        for cn, col in (('quark', '#1f77b4'), ('gluon', '#d62728')):
            h = np.array(R['hist_Q03']['counts'][f'{g}|{cn}'], float); h /= h.sum() * w
            m = A[cn]['mean_Q03']
            ax.step(c, h, where='mid', color=col, lw=1.4, label=f'{cn} jets, mean {m:+.3f}')
            ax.axvline(m, color=col, lw=1, ls=':')
        hq = np.array(R['hist_Q03']['counts'][f'{g}|quark'], float); hq /= hq.sum() * w
        ax.step(-c, hq, where='mid', color='#1f77b4', lw=1.2, ls='--', alpha=0.8,
                label=f'quark jets after charge conjugation E1, mean {-A["quark"]["mean_Q03"]:+.3f}')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(f'{gl}: {A["quark"]["n"] + A["gluon"]["n"]:,} test jets', fontsize=10)
        ax.set_xlabel(r'jet charge $Q_{0.3}=\sum_i q_i\,(p_{T,i}/p_{T,J})^{0.3}$  (units of e)')
        ax.set_xlim(-2, 2); ax.legend(fontsize=7, loc='upper left'); ax.grid(alpha=0.3)
    axs[0].set_ylabel('jets per unit charge (normalised)')
    fig.suptitle('Quark jets in the Z+jet samples carry positive mean jet charge; gluon jets are near zero', fontsize=10.5)
    fig.tight_layout(); p = RES / f'h42_charge_asym{a.tag}{sm}.png'; fig.savefig(p, dpi=140); log('wrote', p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', choices=['run', 'fig', 'merge'], required=True)
    ap.add_argument('--gens', default='py,hw')
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--models', default='')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    {'run': step_run, 'fig': step_fig, 'merge': step_merge}[a.step](a)
