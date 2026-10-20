"""P8v — verification of P8 (where ParT's axis-relative mass forms) on a fresh sample.

Fresh sample: JetClass test_20M WToQQ_102.root + ZToQQ_102.root (P8 used the _101 files), 5,000 random rows each, numpy
default_rng(7); build = part_core.build (eta = jet_eta + sign(jet_eta) * deta). ParT_full only.
Reuses P8's functions (interp.p8_mass_formation_part) for the edits, hooks, patching and DAS.

  1. Unedited accuracy on the P1 sample.
  2. Patching replication (2,000 jets; phi 0.06 and angles x k_match): single sublayers, cumulative blocks 1..l, all 16
     sublayers, donor-resample and random-half controls.
  3. DAS replication: 1-dim jet-mean subspaces at emb, b1, b4, b8 trained on phi shifts ('M2') and on pT scaling ('M0'),
     300 Adam steps (P8: 150); evaluated on phi 0.06, eta 0.06, angles x k_match, pT x 1.25 vs random 1-dim subspaces;
     |cos| between the M2 and M0 directions.
  4. Missing control for P8's "not M2-specific" reading. P8's M2 is the normalised moment S2/S0 (S0 = sum pT, S2 = sum pT dR^2).
     The axis-moment mass mom = sqrt(S0 * S2) (E6; equal to the invariant mass for massless particles about the pT-weighted axis
     in the small-angle limit) changes under pT scaling (dlog mom = log c) as well as under the phi shift (dlog mom = 1/2 dlog M2).
     Per site and direction: response of the jet-mean projection s = mean_p(h_p) . q per unit dlog mom under each edit
     (per-particle route: pair features from the unedited jet, as in DAS). Predictions for ratio R = (ds/dlog mom)[pT x 1.25] /
     (ds/dlog mom)[phi 0.06]: variable = normalised M2 -> R = 0; variable = log mom -> R = 1; variable = invariant mass only ->
     phi response 0 (R unbounded). Same ratio for n98 (full edits). Natural-jet regression of s on log S0 and log(S2/S0) with a
     class term: log mom predicts beta_S0 / (2 beta_M2) = 1.
  5. Substitution redesign at emb, b1, b8 (k = 1): natural donors (same multiplicity batch, same class) of two kinds:
     A = P8's rule (|dm| < 3 GeV, |dlog M2| >= 0.15); B = |dlog mom| >= 0.10 (no mass constraint). Reference: self-interchange
     from the jet's own phi 0.06 counterfactual. Slopes (through 0) of dz97 on the projection difference ds and on dlog mom;
     ratio natural / self; 3 random directions; observed (unpatched) z97 difference donor - recipient.

  ./sync.sh && ./part_job.sh c-p8-verify 'python -m interp.p8v_verify_part'   -> $AI_RESULTS/part/p8_verify.json
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p8_mass_formation_part as p8  # noqa: E402
from interp.part_core import ParTNetwork, sample, kin, build, CL, N98  # noqa: E402
from interp.p3_wz_circuit_part import boot_W, ci, p_two, holm, mcnemar  # noqa: E402

SMOKE = '--smoke' in sys.argv
SEED = 7
PER = 300 if SMOKE else 5000
N_PATCH = 200 if SMOKE else 2000
N_EVAL = 150 if SMOKE else 2000
N_SUB = 300 if SMOKE else 4000
p8.STEPS = 10 if SMOKE else 300
p8.BS = 64 if SMOKE else 128
p8.NB = 100 if SMOKE else 1000
p8.NRAND = 2 if SMOKE else 3
NB = p8.NB
SITES = ['emb', 'b1', 'b4', 'b8']
SUB_SITES = ['emb', 'b1', 'b8']
EDITS = ['phi0.03', 'phi0.06', 'phi0.09', 'eta0.06', 'angk', 'pt0.8', 'pt1.25']
TEST = p8.TEST
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part'
FN = OUT / ('p8_verify_smoke.json' if SMOKE else 'p8_verify.json')
iW, iZ = p8.iW, p8.iZ
log = p8.log


def dump(res):
    with open(FN, 'w') as f: json.dump(p8.strip(res), f, indent=1)


def capture_pp(net, D, names, idx_all, sites):
    """jet-mean residual at `sites` and z97, per-particle route only (pair features of the unedited jet)"""
    n = len(idx_all); pos = np.full(len(D['y']), -1); pos[idx_all] = np.arange(n)
    F = {nm: np.zeros((n, len(sites), p8.C), np.float32) for nm in names}; Z = {nm: np.zeros(n) for nm in names}
    for idx in p8.mbatches(idx_all, D['cnt']):
        rows = pos[idx]; P = int(D['cnt'][idx].max())
        for nm in names:
            x, v, mb = p8.tens(net, D, 'full', nm, idx, P, 'base'); mp = p8.mpn_of(mb); c = mp.sum(0)
            net.clear(); net.want = {p8.SKEY[s] for s in sites}; L, z = net.run(x, v, mb)
            for si, s in enumerate(sites):
                h = net.rec[p8.SKEY[s]]; F[nm][rows, si] = ((h * mp).sum(0) / c).float().cpu().numpy()
            Z[nm][rows] = z[:, N98].double().cpu().numpy()
    net.clear()
    return F, Z


def ratio_boot(Wb, a1, b1, a2, b2):
    """(sum a1 / sum b1) / (sum a2 / sum b2) with paired bootstrap"""
    v = (a1.sum() / b1.sum()) / (a2.sum() / b2.sum())
    bs = ((Wb @ a1) / (Wb @ b1)) / ((Wb @ a2) / (Wb @ b2))
    return dict(v=float(v), ci=ci(bs), p_eq0=p_two(bs, 0.0), p_eq1=p_two(bs, 1.0)), bs


def slope(Wb, y, x):
    s = float((y * x).sum() / (x * x).sum()); bs = (Wb @ (y * x)) / (Wb @ (x * x)); return s, bs


def substitution2(net, D, Qd, Qr, site, sub_idx, rng, sd97, lmom, lM2, mass):
    t = net.t; key = p8.SKEY[site]; n = len(sub_idx); pos = np.full(len(D['y']), -1); pos[sub_idx] = np.arange(n); y = D['y']
    qd = Qd[:, 0]
    VARS = ['das'] + [f'rand{r}' for r in range(len(Qr))]; QV = dict(das=Qd, **{f'rand{r}': Qr[r] for r in range(len(Qr))})
    R = {(s_, v): dict(z=np.full(n, np.nan), c=np.full(n, -1), ds=np.full(n, np.nan), don=np.full(n, -1)) for s_ in 'AB' for v in VARS}
    base = dict(z=np.zeros(n), c=np.zeros(n, int), s=np.zeros(n)); selfi = dict(z=np.zeros(n), ds=np.zeros(n), c=np.zeros(n, int))
    for idx in p8.mbatches(sub_idx, D['cnt']):
        rows = pos[idx]; P = int(D['cnt'][idx].max()); xb, vb, mb = p8.tens(net, D, 'full', 'base', idx, P); mp = p8.mpn_of(mb); nb_ = len(idx)
        net.clear(); net.want = {key}; Lb, zb = net.run(xb, vb, mb); hb = net.rec[key]; net.want = set()
        md = ((hb * mp).sum(0) / mp.sum(0)); sb = (md @ qd).double().cpu().numpy()
        base['z'][rows] = zb[:, N98].double().cpu().numpy(); base['c'][rows] = Lb.argmax(1).cpu().numpy(); base['s'][rows] = sb
        # self-interchange from the jet's own phi 0.06 counterfactual (per-particle route, pair features unedited)
        xd = p8.T(net, D['x']['full']['phi0.06'][idx][:, :, :P]); net.want = {key}; net.run(xd, vb, mb); hd = net.rec[key]; net.want = set()
        net.fn = {key: p8.make_fn('jm', Qd, hd, mp)}; L, z = net.run(xb, vb, mb); net.fn = {}
        selfi['z'][rows] = z[:, N98].double().cpu().numpy(); selfi['c'][rows] = L.argmax(1).cpu().numpy()
        selfi['ds'][rows] = (((hd * mp).sum(0) / mp.sum(0)) @ qd).double().cpu().numpy() - sb
        g = idx
        for s_ in 'AB':
            donor = np.full(nb_, -1)
            for j in range(nb_):
                same = (y[g] == y[g[j]]) & (np.arange(nb_) != j)
                if s_ == 'A': c_ = same & (np.abs(mass[g] - mass[g[j]]) < 3.0) & (np.abs(lM2[g] - lM2[g[j]]) >= 0.15)
                else: c_ = same & (np.abs(lmom[g] - lmom[g[j]]) >= 0.10)
                c_ = np.where(c_)[0]
                if len(c_): donor[j] = rng.choice(c_)
            ok = donor >= 0
            if not ok.any(): continue
            dn = p8.T(net, np.where(ok, donor, np.arange(nb_)).astype(np.int64))
            for v in VARS:
                net.fn = {key: p8.make_fn('jm', QV[v], hb, mp, donor=dn)}; L, z = net.run(xb, vb, mb); net.fn = {}
                q = QV[v][:, 0]; sv = (md @ q).double().cpu().numpy()
                r = R[(s_, v)]; rr = rows[ok]
                r['z'][rr] = z[ok][:, N98].double().cpu().numpy(); r['c'][rr] = L[ok].argmax(1).cpu().numpy()
                r['ds'][rr] = sv[donor[ok]] - sv[ok]; r['don'][rr] = g[donor[ok]]
    net.clear()
    out = dict(site=site, sets={})
    rid_all = sub_idx
    for s_ in 'AB':
        has = R[(s_, 'das')]['don'] >= 0; rid = rid_all[has]; did = R[(s_, 'das')]['don'][has]; nn = int(has.sum())
        Wb = boot_W(nn, NB, np.random.default_rng(SEED + 5)).astype(np.float64)
        dmom = lmom[did] - lmom[rid]; dM2 = lM2[did] - lM2[rid]; dm = mass[did] - mass[rid]; yy = y[rid]
        dz0 = (base['z'][has]) ; c0 = base['c'][has]
        # self reference on the same recipients
        dz_self = (selfi['z'][has] - dz0) / sd97; ds_self = selfi['ds'][has]; dmom_phi = (D['obs']['phi0.06']['log_mom'] - D['obs']['base']['log_mom'])[rid]
        s_self_ds, b_self_ds = slope(Wb, dz_self, ds_self); s_self_mom, b_self_mom = slope(Wb, dz_self, dmom_phi)
        o = dict(N=nn, donor=dict(mean_abs_dlog_mom=float(np.abs(dmom).mean()), mean_abs_dlogM2=float(np.abs(dM2).mean()), mean_abs_dm=float(np.abs(dm).mean()),
                                  corr_dlogmom_dlogM2=float(np.corrcoef(dmom, dM2)[0, 1]), sd_dlog_mom=float(dmom.std())),
                 self_phi=dict(slope_dz_per_ds=s_self_ds, ci=ci(b_self_ds), slope_dz_per_dlogmom=s_self_mom, ci_mom=ci(b_self_mom),
                               mean_ds=float(ds_self.mean()), mean_dz=float(dz_self.mean())),
                 variants={})
        # observed (unpatched) z97 difference donor - recipient
        pos_d = pos[did]; obs_dz = (base['z'][pos_d] - dz0) / sd97 if (pos_d >= 0).all() else None
        if obs_dz is not None:
            so, bo = slope(Wb, obs_dz, dmom); o['observed_z97_diff'] = dict(slope_per_dlogmom=so, ci=ci(bo))
            so2, bo2 = slope(Wb, obs_dz, dM2); o['observed_z97_diff']['slope_per_dlogM2'] = so2; o['observed_z97_diff']['ci_M2'] = ci(bo2)
        for v in VARS:
            r = R[(s_, v)]; dz = (r['z'][has] - dz0) / sd97; ds = r['ds'][has]; c1 = r['c'][has]
            sds, bds = slope(Wb, dz, ds); smom, bmom = slope(Wb, dz, dmom); sM2, bM2 = slope(Wb, dz, dM2)
            e = dict(slope_dz_per_ds=sds, ci=ci(bds), slope_dz_per_dlogmom=smom, ci_mom=ci(bmom), slope_dz_per_dlogM2=sM2, ci_M2=ci(bM2),
                     corr_ds_dlogmom=float(np.corrcoef(ds, dmom)[0, 1]), corr_ds_dlogM2=float(np.corrcoef(ds, dM2)[0, 1]),
                     changed=float((c1 != c0).mean()), mean_abs_ds=float(np.abs(ds).mean()),
                     mcnemar=mcnemar(c0 == yy, c1 == yy))
            if v == 'das':
                rb = bds / b_self_ds; e['ratio_ds_slope_to_self'] = dict(v=float(sds / s_self_ds), ci=ci(rb), p_eq1=p_two(rb, 1.0), p_eq0=p_two(rb, 0.0))
                rm = bmom / b_self_mom; e['ratio_mom_slope_to_self'] = dict(v=float(smom / s_self_mom), ci=ci(rm), p_eq1=p_two(rm, 1.0), p_eq0=p_two(rm, 0.0))
                o['self_phi']['mean_abs_ds_ratio_nat_over_self'] = float(np.abs(ds).mean() / np.abs(ds_self).mean())
            o['variants'][v] = e
        out['sets'][s_] = o
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    res = dict(node='P8v', seed=SEED, smoke=SMOKE, files=['WToQQ_102.root', 'ZToQQ_102.root'],
               sizes=dict(PER=PER, N_PATCH=N_PATCH, N_EVAL=N_EVAL, N_SUB=N_SUB, NB=NB, STEPS=p8.STEPS, BS=p8.BS, NRAND=p8.NRAND))
    JW, rW = p8.load_rows(TEST / 'WToQQ_102.root', PER, rng); JZ, rZ = p8.load_rows(TEST / 'ZToQQ_102.root', PER, rng)
    J0 = p8.cat([JW, JZ]); res['rows_head'] = dict(W=rW[:5].tolist(), Z=rZ[:5].tolist()); del JW, JZ
    D = p8.prepare(J0, EDITS, ['full'], ['phi0.06', 'eta0.06', 'angk', 'pt1.25']); del J0
    N = len(D['y']); y = D['y']; ob = D['obs']
    lmom = ob['base']['log_mom']; lM2 = ob['base']['log_M2']; mass = ob['base']['m']; lS0 = lmom - 0.5 * lM2
    res['k_match'] = D['k_match']
    res['truth'] = {e: dict(dlog_m=float(np.mean(np.log(ob[e]['m'] / ob['base']['m']))), dlog_M2=float(np.mean(ob[e]['log_M2'] - lM2)),
                            dlog_mom=float(np.mean(ob[e]['log_mom'] - lmom))) for e in EDITS}
    log('truth', res['truth'])
    perm = np.random.default_rng(SEED).permutation(N); tr, te = np.sort(perm[:N // 2]), np.sort(perm[N // 2:])
    pn = ParTNetwork('full'); net = p8.Net(pn)
    # ---- 1. accuracy
    P1 = sample(30 if SMOKE else 1000, 3000); a, b, c, d = kin(P1); P1b = build(P1, a, b, c, d)
    res['acc_P1_full'] = float((pn.run(P1b)['logits'].argmax(1) == P1b['label'].astype(int)).mean()); del P1, P1b
    allidx = np.arange(N)
    _, Zf, Lf = p8.capture(net, D, 'full', ['base', 'phi0.06', 'eta0.06', 'angk', 'pt1.25'], allidx)
    sd97 = float(Zf['base'].std()); res['n98_sd'] = sd97
    res['acc_WZ_10class'] = float((Lf['base'].argmax(1) == y).mean())
    res['full_edits'] = {}
    Wn = boot_W(N, NB, np.random.default_rng(SEED + 1)).astype(np.float64)
    for e in ('phi0.06', 'eta0.06', 'angk', 'pt1.25'):
        dz = (Zf[e] - Zf['base']) / sd97; dmo = ob[e]['log_mom'] - lmom
        res['full_edits'][e] = dict(dz97=float(dz.mean()), dz97_ci=ci((Wn @ dz) / Wn.sum(1)), W_to_Z_base=float((Lf['base'].argmax(1)[y == iW] == iZ).mean()),
                                    W_to_Z=float((Lf[e].argmax(1)[y == iW] == iZ).mean()), dz97_per_dlogmom=float(dz.mean() / dmo.mean()))
    dzp = (Zf['phi0.06'] - Zf['base']) / sd97; dmp = ob['phi0.06']['log_mom'] - lmom
    for e in ('pt1.25', 'angk', 'eta0.06'):
        dz = (Zf[e] - Zf['base']) / sd97; dmo = ob[e]['log_mom'] - lmom
        res['full_edits'][e]['n98_ratio_per_dlogmom_to_phi'], _ = ratio_boot(Wn, dz, dmo, dzp, dmp)
    log(f'acc P1 {res["acc_P1_full"]:.4f}; WZ 10-class {res["acc_WZ_10class"]:.4f}; full edits', res['full_edits'])
    dump(res); del Zf, Lf
    # ---- 2. patching replication
    t1 = time.time(); pidx = np.sort(rng.choice(N, min(N_PATCH, N), replace=False))
    pf = p8.patching(net, D, 'full', pidx, ['phi0.06', 'angk'], rng, True, sd97)
    keep = {}
    for e, o in pf['edits'].items():
        V = o['variants']
        keep[e] = dict(edit=o['edit'], single={k: V[k]['n98'] for k in V if k[:5] in ('den_a', 'den_m', 'res_a', 'res_m', 'sub_a', 'sub_m') and k[5:].isdigit()},
                       cumulative={k: V[k]['n98'] for k in V if k.startswith('den_upto') or k in ('den_allA', 'den_allM', 'den_allsub', 'noi_allsub', 'noi_allA', 'noi_allM')})
        sing = {k: d['v'] for k, d in keep[e]['single'].items() if k[:5] in ('den_a', 'den_m')}
        keep[e]['max_single'] = max(sing.items(), key=lambda kv: kv[1])
        keep[e]['max_single_resample'] = max(((k, d['v']) for k, d in keep[e]['single'].items() if k[:3] == 'res'), key=lambda kv: kv[1])
        log(f'patching {e}: max single {keep[e]["max_single"]}, allsub {V["den_allsub"]["n98"]["v"]:.3f} {V["den_allsub"]["n98"]["ci"]}, '
            f'upto ' + ', '.join(f'{V[f"den_upto{l}"]["n98"]["v"]:.3f}' for l in range(1, 8)))
    res['patching'] = dict(N=pf['N'], edits=keep, holm_n_tests=pf['holm_n_tests']); log(f'(2) patching {time.time() - t1:.0f} s'); dump(res)
    # ---- 3. DAS replication (k = 1, 300 steps)
    t1 = time.time(); ev = np.sort(rng.choice(te, min(N_EVAL, len(te)), replace=False))
    plan = [('M2', s, 'jm', 1) for s in SITES] + [('M0', s, 'jm', 1) for s in SITES]
    das, Qs = p8.das_block(net, D, 'full', plan, ['phi0.06', 'eta0.06', 'angk', 'pt1.25'], tr, ev, rng, 'verify')
    res['das'] = das
    res['cos_M2_M0'] = {s: float(abs((Qs[f'M2|{s}|jm|1'][:, 0] * Qs[f'M0|{s}|jm|1'][:, 0]).sum())) for s in SITES}
    S_ = das['summary']
    for s in SITES:
        log(f'DAS {s}: M2 ' + ' '.join(f'{cf} {S_[f"M2|{s}|jm|1"][cf]["rec"]:.3f}' for cf in ('phi0.06', 'eta0.06', 'angk', 'pt1.25'))
            + ' | M0 ' + ' '.join(f'{cf} {S_[f"M0|{s}|jm|1"][cf]["rec"]:.3f}' for cf in ('phi0.06', 'pt1.25'))
            + f' | rand {S_[f"rand|{s}|jm|1"]["phi0.06"]["rec"]:.3f} | cos {res["cos_M2_M0"][s]:.3f}')
    log(f'(3) DAS {time.time() - t1:.0f} s'); dump(res)
    # ---- 4. projection responses per unit dlog mom (per-particle route) + natural regression
    t1 = time.time(); F, _ = capture_pp(net, D, ['base'] + EDITS, allidx, SITES)
    Wb = boot_W(N, NB, np.random.default_rng(SEED + 2)).astype(np.float64)
    res['projection'] = {}; ps_R = []; keys_R = []
    X = np.stack([np.ones(N), (y == iZ).astype(float), lS0, lM2], 1)
    for si, s in enumerate(SITES):
        res['projection'][s] = {}
        for var in ('M2', 'M0'):
            q = Qs[f'{var}|{s}|jm|1'][:, 0].double().cpu().numpy()
            s0 = F['base'][:, si].astype(np.float64) @ q
            o = dict(edits={})
            ds_phi = F['phi0.06'][:, si].astype(np.float64) @ q - s0; dm_phi = ob['phi0.06']['log_mom'] - lmom
            for e in EDITS:
                ds = F[e][:, si].astype(np.float64) @ q - s0; dmo = ob[e]['log_mom'] - lmom
                o['edits'][e] = dict(mean_ds=float(ds.mean()), mean_ds_ci=ci((Wb @ ds) / Wb.sum(1)), dlog_mom=float(dmo.mean()),
                                     ds_per_dlogmom=float(ds.mean() / dmo.mean()) if abs(dmo.mean()) > 1e-9 else None)
                if e != 'phi0.06' and abs(dmo.mean()) > 1e-9:
                    o['edits'][e]['ratio_per_dlogmom_to_phi'], _ = ratio_boot(Wb, ds, dmo, ds_phi, dm_phi)
            # natural regression s ~ 1 + isZ + log S0 + log(S2/S0)
            beta = np.linalg.lstsq(X, s0, rcond=None)[0]; resid = s0 - X @ beta; r2 = 1 - resid.var() / s0.var()
            bsr = []
            for b in range(200):
                ii = np.random.default_rng(SEED + 100 + b).integers(0, N, N); bb = np.linalg.lstsq(X[ii], s0[ii], rcond=None)[0]; bsr.append(bb[2] / (2 * bb[3]))
            r2s = {}
            for nm_, xv in (('log_m', np.log(mass)), ('log_mom', lmom), ('log_M2', lM2), ('log_S0', lS0)):
                Xs = np.stack([np.ones(N), (y == iZ).astype(float), xv], 1); bb = np.linalg.lstsq(Xs, s0, rcond=None)[0]; r2s[nm_] = float(1 - (s0 - Xs @ bb).var() / s0.var())
            o['natural'] = dict(beta_logS0=float(beta[2]), beta_logM2=float(beta[3]), r2=float(r2), ratio_S0_over_2M2=float(beta[2] / (2 * beta[3])),
                                ratio_ci=ci(bsr), r2_single=r2s,
                                edit_beta_logS0=float(o['edits']['pt1.25']['mean_ds'] / np.log(1.25)),
                                edit_beta_logM2=float(o['edits']['phi0.06']['mean_ds'] / np.mean(ob['phi0.06']['log_M2'] - lM2)))
            res['projection'][s][var] = o
            R_ = o['edits']['pt1.25']['ratio_per_dlogmom_to_phi']; keys_R.append(f'{s}|{var}'); ps_R.append(R_['p_eq1'])
            log(f'proj {s} {var}: ds/dlogmom ' + ', '.join(f'{e} {o["edits"][e]["ds_per_dlogmom"]:+.3f}' for e in EDITS)
                + f' | pt/phi R {R_["v"]:.3f} {R_["ci"]} | natural S0/(2 M2) {o["natural"]["ratio_S0_over_2M2"]:.3f} {o["natural"]["ratio_ci"]} R2 {o["natural"]["r2"]:.3f}')
    res['projection_holm_peq1'] = dict(zip(keys_R, holm(ps_R)))
    del F; log(f'(4) projection {time.time() - t1:.0f} s'); dump(res)
    # ---- 5. substitution redesign
    t1 = time.time(); sidx = np.sort(rng.choice(te, min(N_SUB, len(te)), replace=False)); res['substitution'] = {}
    for s in SUB_SITES:
        Qr = [p8.rand_Q(net, 1, rng) for _ in range(p8.NRAND)]
        o = substitution2(net, D, Qs[f'M2|{s}|jm|1'], Qr, s, sidx, rng, sd97, lmom, lM2, mass); res['substitution'][s] = o
        for set_, d in o['sets'].items():
            v = d['variants']['das']
            log(f'sub {s} set {set_} (N {d["N"]}, |dlogmom| {d["donor"]["mean_abs_dlog_mom"]:.3f}, |dlogM2| {d["donor"]["mean_abs_dlogM2"]:.3f}): '
                f'dz/ds nat {v["slope_dz_per_ds"]:+.3f} self {d["self_phi"]["slope_dz_per_ds"]:+.3f} ratio {v["ratio_ds_slope_to_self"]["v"]:.3f} {v["ratio_ds_slope_to_self"]["ci"]}; '
                f'dz/dlogmom nat {v["slope_dz_per_dlogmom"]:+.3f} self {d["self_phi"]["slope_dz_per_dlogmom"]:+.3f}; dz/dlogM2 {v["slope_dz_per_dlogM2"]:+.3f}; '
                f'corr(ds, dlogmom) {v["corr_ds_dlogmom"]:.3f}; changed {v["changed"]:.3f}; rand dz/dlogmom '
                + ', '.join(f'{d["variants"][f"rand{r}"]["slope_dz_per_dlogmom"]:+.3f}' for r in range(p8.NRAND))
                + (f'; observed z97 diff per dlogmom {d["observed_z97_diff"]["slope_per_dlogmom"]:+.3f}' if 'observed_z97_diff' in d else ''))
        dump(res)
    log(f'(5) substitution {time.time() - t1:.0f} s')
    res['elapsed_s'] = time.time() - p8.T0; dump(res); log(f'wrote {FN}'); log('P8v DONE')


if __name__ == '__main__':
    main()
