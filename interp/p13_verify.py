"""P13 verify: rerun the P13 feature-circuit test on fresh jets with the trained P13 dictionaries, plus missing controls.

Reuses the P13 dictionaries (/j-jepa-vol/aaron-interp/results/part/p13_dicts.pt) and P13's hooked runner / mask hook.
Fresh samples (seed 29, random rows, never first-k; files not used by P13 or P1-P9):
  ALL10 : test_20M <sample>_110.root, NALL random rows per class -> 10-class accuracy, all-12 splice, 10-class circuit read
  DISC  : WToQQ/ZToQQ_108.root, NDISC each -> fresh IG ranking, m_A scan, steering calibration
  EVAL  : WToQQ/ZToQQ_109.root, NEVAL each -> circuits kept alone (fresh ranking and the original P13 ranking)
Added controls:
  (a) random equal-size feature sets also at K = 1,000 and 2,000 (P13 ran K <= 200 only)
  (b) single-feature dose: the top m_A feature steered by 0.16 ... 8 s.d. toward higher m_A
  (c) joint steering of all live features of one class-level site (C0, R0, C1, R1) by beta_i * dlog m_A(phi edit), beta_i from
      natural same-class pT-matched pairs; control = beta permuted among the site's live features (NPERM permutations)
  ./part_job.sh c-p13-verify 'python -m interp.p13_verify'
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.p13_transcoder_part import (make_dict_cls, Runner, Rec, Splice, MaskH, Steer, Hook, prepset, load_rows, to_dev,  # noqa: E402
                                        jet_batches, observables, bh, SITES, DTC, DSAE, K_TOP, TEST, OUT)
from interp.p9_eap_wz_part import boot_W, ci, mcnemar, clean, donors_B, donors_U, confusion  # noqa: E402

SMOKE = '--smoke' in sys.argv
NALL = 60 if SMOKE else 500; NDISC = 100 if SMOKE else 1500; NEVAL = 150 if SMOKE else 5000
NRAND = 3 if SMOKE else 40; NPERM = 3 if SMOKE else 20; NB = 200 if SMOKE else 1000; BATCH = 64
KGRID = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]; KRAND = [20, 50, 200, 1000, 2000]; K10 = [50, 1000, 5000]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]; SEED = 29
FN = OUT / f'p13_verify{"_smoke" if SMOKE else ""}.json'; T0 = time.time()


def log(*a): print(f'[{time.time() - T0:6.0f} s]', *a, flush=True)


class MultiSteer(Hook):
    """add delta (N, D) of feature values at site s (class token): y + s_out * delta @ W_dec"""
    def __init__(self, D, s, delta): self.D, self.s, self.delta = D, s, delta

    def _do(self, s, y):
        if s != self.s: return y
        Dm = self.D[s]; return y + Dm.s_out * (self.delta @ Dm.W_dec)[None]

    def mlp(self, s, u, y, hid): return self._do(s, y)
    def resid(self, s, r): return self._do(s, r)


def main():
    import torch as t
    from interp.part_core import ParTNetwork, inputs, sub, cat, kin, build, moments, CL, PRE
    OUT.mkdir(parents=True, exist_ok=True); iW, iZ = CL.index('Wqq'), CL.index('Zqq'); res = dict(smoke=SMOKE, seed=SEED)

    def dump(): FN.write_text(json.dumps(clean(res), indent=1))
    net = ParTNetwork('full'); model = net.model; dev = net.device
    for p in model.parameters(): p.requires_grad_(False)
    run = Runner(model); Dict = make_dict_cls()
    blob = t.load(OUT / 'p13_dicts.pt', map_location=dev); D = {}
    for s in SITES:
        D[s] = Dict(128, DTC if s[0] in 'PC' else DSAE, K_TOP, s[0] in 'PC').to(dev); D[s].load_state_dict(blob['dicts'][s])
        for p in D[s].parameters(): p.requires_grad_(False)
    orig = np.load(OUT / 'p13_scores.npz'); log('dictionaries loaded')

    # ------------------------------------------------------------------ samples
    rng = np.random.default_rng(SEED); files = {}; pA = []
    for c in CL:
        J, inf = load_rows(TEST / f'{PRE[c]}_110.root', NALL, rng); pA.append(J); files[f'all_{c}'] = inf
    JA, yA = prepset(pA); del pA; pD, pE = [], []
    for c in ('Wqq', 'Zqq'):
        J, inf = load_rows(TEST / f'{PRE[c]}_108.root', NDISC, rng); pD.append(J); files[f'disc_{c}'] = inf
        J, inf = load_rows(TEST / f'{PRE[c]}_109.root', NEVAL, rng); pE.append(J); files[f'eval_{c}'] = inf
    JD, yD = prepset(pD); JE, yE = prepset(pE); del pD, pE; res['samples'] = files
    log(f'samples all {len(yA)} disc {len(yD)} eval {len(yE)}')

    def logits(J, H=None, idx=None):
        idx = np.arange(len(J['mask'])) if idx is None else np.asarray(idx); L = np.zeros((len(idx), 10), np.float32)
        for bb in jet_batches(J['mask'][idx].sum(1), 64):
            x, v, mk = to_dev(t, dev, inputs(sub(J, idx[bb]), 'full'))
            with t.no_grad(): L[bb] = run(x, v, mk, H).float().cpu().numpy()
        return L

    # ------------------------------------------------------------------ 1. unedited + splice
    LA = logits(JA); okA = LA.argmax(1) == yA; WbA = boot_W(len(yA), NB, np.random.default_rng(1))
    x, v, mk = to_dev(t, dev, inputs(sub(JA, np.arange(0, len(yA), max(1, len(yA) // 64))[:64]), 'full'))
    with t.no_grad(): dmax = float((model(x, v, mk) - run(x, v, mk)).abs().max())
    LS = logits(JA, Splice(D, SITES)); okS = LS.argmax(1) == yA
    res['unedited'] = dict(acc10=float(okA.mean()), acc10_ci=ci(WbA @ okA / len(yA)), runner_vs_weaver=dmax,
                           splice_all_acc10=float(okS.mean()), splice_drop_ci=ci((WbA @ okA - WbA @ okS) / len(yA)), splice_mcnemar=mcnemar(okA, okS))
    log('unedited', res['unedited']); dump()

    # ------------------------------------------------------------------ 2. fresh IG ranking + live mask
    rngd = np.random.default_rng(SEED); donD = donors_B(JD['jet_pt'], yD, iW, iZ, rngd); donE = donors_B(JE['jet_pt'], yE, iW, iZ, rngd)
    donA = donors_U(len(yA), rngd)

    def pair_batches(J, idx, don, size=BATCH):
        idx = np.asarray(idx); m = np.maximum(J['mask'][idx].sum(1), J['mask'][don[idx]].sum(1))
        for bb in jet_batches(m, size):
            bi = idx[bb]; x, v, mk = inputs(cat([sub(J, bi), sub(J, don[bi])]), 'full'); n = len(bi)
            xs = to_dev(t, dev, (x, v, mk)); yield bb, [a[:n] for a in xs], [a[n:] for a in xs]

    def mvec(val, grad=False):
        return {s: (t.full((D[s].D,), float(val), device=dev, requires_grad=grad), t.tensor(float(val), device=dev, requires_grad=grad)) for s in SITES}

    scores = {s: np.zeros(D[s].D) for s in SITES}; serr = {s: 0.0 for s in SITES}; fired = {s: np.zeros(D[s].D, bool) for s in SITES}
    sD = np.where(yD == iW, 1.0, -1.0); H = MaskH(D); Dt = [0.0, 0.0]
    for bb, cl, dn in pair_batches(JD, np.arange(len(yD)), donD, 48):
        s_ = t.from_numpy(sD[bb]).float().to(dev); H.mode = 'donor'
        with t.no_grad():
            Ld = run(*dn, H); R = Rec(SITES); Lc = run(*cl, R)
            real = cl[2][:, 0].bool().T
            for s in SITES:
                u = R.d[s][0]; a = D[s].enc(u / D[s].s_in)
                a = a[real] if s[0] == 'P' else a[0]; fired[s] |= (a > 0).any(0).cpu().numpy()
        Dt[0] += float((s_ * (Lc[:, iW] - Lc[:, iZ])).sum()); Dt[1] += float((s_ * (Ld[:, iW] - Ld[:, iZ])).sum()); H.mode = 'clean'
        for al in ALPHAS:
            mm = mvec(al, True); H.spec = dict(m=mm); L = run(*cl, H); Dm_ = (s_ * (L[:, iW] - L[:, iZ])).sum()
            gr = t.autograd.grad(Dm_, [q for s in SITES for q in mm[s]])
            for i, s in enumerate(SITES): scores[s] += gr[2 * i].cpu().numpy() / len(ALPHAS); serr[s] += float(gr[2 * i + 1]) / len(ALPHAS)
    allf = [(s, i, scores[s][i]) for s in SITES for i in range(D[s].D)]; tot_abs = sum(abs(r[2]) for r in allf) + sum(abs(v) for v in serr.values())
    rank = sorted([r for r in allf if fired[r[0]][r[1]]], key=lambda r: -abs(r[2]))
    rank_o = sorted([(s, i, float(orig[s][i])) for s in SITES for i in range(D[s].D) if orig[s][i] != 0], key=lambda r: -abs(r[2]))
    from scipy.stats import spearmanr
    va = np.concatenate([scores[s] for s in SITES]); vo = np.concatenate([orig[s] for s in SITES])
    top = lambda r, k: {(a, b) for a, b, _ in r[:k]}
    res['attribution'] = dict(completeness=(sum(r[2] for r in allf) + sum(serr.values())) / (Dt[0] - Dt[1]), error_abs_share=sum(abs(v) for v in serr.values()) / tot_abs,
                              top50_abs_share=sum(abs(r[2]) for r in rank[:50]) / tot_abs, n_live=len(rank),
                              spearman_score_vs_P13=float(spearmanr(va, vo)[0]), pearson_score_vs_P13=float(np.corrcoef(va, vo)[0, 1]),
                              topk_overlap_vs_P13={str(k): len(top(rank, k) & top(rank_o, k)) / k for k in (10, 50, 200, 1000)},
                              top50_sites={s: sum(1 for r in rank[:50] if r[0] == s) for s in SITES},
                              top10=[dict(feature=f'{s}#{i}', score_per_jet=v / len(yD)) for s, i, v in rank[:10]])
    log('attribution', {k: v for k, v in res['attribution'].items() if k != 'top10'}); dump()

    # ------------------------------------------------------------------ 3. circuits kept alone
    def spec_of(feats, err=0.0, **kw):
        m = {s: (t.zeros(D[s].D, device=dev), t.tensor(float(err), device=dev)) for s in SITES}
        for s, i in feats: m[s][0][i] = 1.0
        return dict(m=m, **kw)

    def run_specs(J, idx, don, specs, check=False):
        idx = np.asarray(idx); out = {k: np.zeros((len(idx), 10), np.float32) for k in list(specs) + ['clean', 'corr']}; H = MaskH(D); chk = [0.0, 0.0]
        with t.no_grad():
            for bb, cl, dn in pair_batches(J, idx, don):
                H.mode = 'donor'; Ld = run(*dn, H); Lc = run(*cl); H.mode = 'clean'; out['clean'][bb] = Lc.cpu().numpy(); out['corr'][bb] = Ld.cpu().numpy()
                if check:
                    H.spec = dict(m=mvec(1.0)); chk[0] = max(chk[0], float((run(*cl, H) - Lc).abs().max()))
                    H.spec = dict(m=mvec(0.0)); chk[1] = max(chk[1], float((run(*cl, H) - Ld).abs().max()))
                for k, sp in specs.items(): H.spec = sp; out[k][bb] = run(*cl, H).cpu().numpy()
        if check: out['_check'] = dict(all_kept_vs_clean=chk[0], none_kept_vs_donor=chk[1])
        return out

    sE = np.where(yE == iW, 1.0, -1.0); WbE = boot_W(len(yE), NB, np.random.default_rng(2))

    def summ(L, Lc, Lco, y, s, Wb):
        d = L[:, iW] - L[:, iZ]; dc = Lc[:, iW] - Lc[:, iZ]; dco = Lco[:, iW] - Lco[:, iZ]
        num = s * (d - dco); den = s * (dc - dco); ok = np.where(d > 0, iW, iZ) == y; ok0 = np.where(dc > 0, iW, iZ) == y
        return dict(F=float(num.sum() / den.sum()), wz_acc=float(ok.mean()), wz_acc_ci=ci((Wb @ ok) / len(y)), mcnemar_vs_full=mcnemar(ok0, ok),
                    agree_with_full=float((np.sign(d) == np.sign(dc)).mean()))
    specs = {}
    for K in KGRID:
        f = [(s, i) for s, i, _ in rank[:K]]; fo = [(s, i) for s, i, _ in rank_o[:K]]
        specs[f'main_K{K}'] = spec_of(f); specs[f'origrank_main_K{K}'] = spec_of(fo); specs[f'errlive_K{K}'] = spec_of(f, 1.0)
        specs[f'strict_K{K}'] = spec_of(f, 0.0, skip_live=False, attn_res=True)
    specs['main_Kall_live'] = spec_of([(s, i) for s, i, _ in rank])
    specs['strict_all_features_and_errors'] = spec_of([(s, i) for s in SITES for i in range(D[s].D)], 1.0, skip_live=False, attn_res=True)
    OE = run_specs(JE, np.arange(len(yE)), donE, specs, check=True)
    okc = np.where(OE['clean'][:, iW] > OE['clean'][:, iZ], iW, iZ) == yE
    circ = dict(unedited_wz_acc=float(okc.mean()), unedited_wz_acc_ci=ci(WbE @ okc / len(yE)),
                unedited_acc10_on_eval=float((OE['clean'].argmax(1) == yE).mean()),
                donor_wz_acc=float((np.where(OE['corr'][:, iW] > OE['corr'][:, iZ], iW, iZ) == yE).mean()), exactness=OE['_check'],
                K={k: summ(OE[k], OE['clean'], OE['corr'], yE, sE, WbE) for k in specs})
    for k in circ['K']:
        if k.startswith('main_K') and k[6:].isdigit(): circ['K'][k]['site_counts'] = {s: sum(1 for r in rank[:int(k[6:])] if r[0] == s) for s in SITES}
    res['circuits'] = circ
    log(f'circuits: unedited {circ["unedited_wz_acc"]:.4f} donor {circ["donor_wz_acc"]:.4f} {OE["_check"]}')
    for K in KGRID: log(f'  K {K}: ' + ' | '.join(f'{v} {circ["K"][f"{v}_K{K}"]["wz_acc"]:.3f}' for v in ('main', 'origrank_main', 'errlive', 'strict')))
    dump()

    # random sets, matched per-site counts, K up to 2,000
    sub_idx = np.arange(0, len(yE), 5); rngr = np.random.default_rng(SEED + 1); livelist = {s: np.where(fired[s])[0] for s in SITES}; rnd = {}
    ys, ss_ = yE[sub_idx], sE[sub_idx]; Wb = boot_W(len(ys), NB, np.random.default_rng(4))
    for K in KRAND:
        cnt = circ['K'][f'main_K{K}']['site_counts']; sp = {'circ': specs[f'main_K{K}']}
        for j in range(NRAND):
            sp[f'r{j}'] = spec_of([(s, int(i)) for s in SITES for i in rngr.choice(livelist[s], min(cnt[s], len(livelist[s])), replace=False)])
        O = run_specs(JE, sub_idx, donE, sp); cs = summ(O['circ'], O['clean'], O['corr'], ys, ss_, Wb)
        ra = np.array([summ(O[f'r{j}'], O['clean'], O['corr'], ys, ss_, Wb)['wz_acc'] for j in range(NRAND)])
        rnd[str(K)] = dict(n_jets=len(ys), circuit_wz_acc=cs['wz_acc'], circuit_ci=cs['wz_acc_ci'], random_mean=float(ra.mean()), random_q=ci(ra),
                           random_max=float(ra.max()), p_perm=float((1 + (ra >= cs['wz_acc']).sum()) / (NRAND + 1)))
        log(f'  random K {K}: circuit {cs["wz_acc"]:.3f} random mean {ra.mean():.3f} max {ra.max():.3f}')
    res['circuits']['random_sets'] = rnd; dump()
    # 10-class read
    sp = {f'main_K{K}': specs[f'main_K{K}'] for K in K10}; O = run_specs(JA, np.arange(len(yA)), donA, sp); ten = {}
    for k in sp:
        okk = O[k].argmax(1) == yA; ten[k] = dict(acc10=float(okk.mean()), acc10_ci=ci(WbA @ okk / len(yA)), follows_donor=float((O[k].argmax(1) == yA[donA]).mean()))
    res['circuits']['ten_class'] = ten; log('10-class', ten); dump()

    # ------------------------------------------------------------------ 4. m_A steering: dose and joint
    OD = observables(JD); lmD = OD['log_mA']; CS = ('C0', 'R0', 'C1', 'R1'); acts = {s: [] for s in CS}; ordD = []
    for bb in jet_batches(JD['mask'].sum(1), 64):
        x, v, mk = to_dev(t, dev, inputs(sub(JD, bb), 'full')); R = Rec(CS)
        with t.no_grad():
            run(x, v, mk, R)
            for s in CS: acts[s].append(D[s].enc(R.d[s][0][0] / D[s].s_in).cpu().numpy())
        ordD.append(bb)
    inv = np.argsort(np.concatenate(ordD)); acts = {s: np.concatenate(a)[inv] for s, a in acts.items()}
    from scipy.stats import pearsonr
    scan = [(s, int(i), *map(float, pearsonr(A[:, i], lmD))) for s, A in acts.items() for i in np.where(A.std(0) > 0)[0]]
    q = bh([r[3] for r in scan]); j = int(np.argmax([abs(r[2]) for r in scan])); fs, fi, fr, _ = scan[j]
    rngp = np.random.default_rng(31); part = np.zeros(len(yD), int)
    for c in (iW, iZ):
        ix = np.where(yD == c)[0]; o = ix[np.argsort(JD['jet_pt'][ix])]
        for k, i in enumerate(o): part[i] = rngp.choice(np.concatenate([o[max(0, k - 3):k], o[k + 1:k + 4]]))
    dl = lmD[part] - lmD; beta = {s: ((acts[s][part] - acts[s]) * dl[:, None]).sum(0) / (dl * dl).sum() for s in CS}
    sig = {s: np.zeros(D[s].D, bool) for s in CS}
    for (s, i, r, p), qq in zip(scan, q): sig[s][i] = qq < 0.05
    st = dict(scan_n=len(scan), scan_bh_sig=int((q < 0.05).sum()), top_feature=f'{fs}#{fi}', top_r=fr, top5=[f'{a}#{b} {c:+.3f}' for a, b, c, _ in sorted(scan, key=lambda r: -abs(r[2]))[:5]],
              p13_feature_C0_831_r=next((r for a, b, r, _ in scan if a == 'C0' and b == 831), None))
    wi = np.where(yE == iW)[0]; JW = sub(JE, wi); pt, de, dp, ms = kin(JW); JWp = build(JW, pt, de, dp + 0.06 * (pt > 0), ms)
    dlm = 0.5 * np.log(moments(*kin(JWp)[:3])['M0M2']) - 0.5 * np.log(moments(*kin(JW)[:3])['M0M2']); nW = len(wi); WbW = boot_W(nW, NB, np.random.default_rng(5))
    Lb = logits(JW); Lp = logits(JWp); flow = lambda L: (L.argmax(1) == iZ).astype(float); fb = flow(Lb); fphi = flow(Lp) - fb
    st.update(n_W=nW, mean_dlog_mA_phi=float(dlm.mean()), W_to_Z_base=float(fb.mean()), flow_phi=float(fphi.mean()), flow_phi_ci=ci(WbW @ fphi / nW),
              unedited_W_acc10=float((Lb.argmax(1) == iW).mean()))

    def steer_multi(J, s, delta):
        out = np.zeros((len(J['mask']), 10), np.float32)
        for bb in jet_batches(J['mask'].sum(1), 64):
            x, v, mk = to_dev(t, dev, inputs(sub(J, bb), 'full'))
            with t.no_grad(): out[bb] = run(x, v, mk, MultiSteer(D, s, t.from_numpy(np.ascontiguousarray(delta[bb], np.float32)).to(dev))).cpu().numpy()
        return out

    dd = lambda L: (L[:, iZ] - L[:, iW]) - (Lb[:, iZ] - Lb[:, iW]); ddphi = dd(Lp)
    # natural-pair calibration of the read-out: slope of (l_Z - l_W) on log m_A over same-class pT-matched pairs (DISC)
    LDn = logits(JD); dZW = LDn[:, iZ] - LDn[:, iW]; ddn = dZW[part] - dZW; slope = float((ddn * dl).sum() / (dl * dl).sum())
    bsl = []
    for _ in range(NB):
        b_ = rngp.integers(0, len(dl), len(dl)); bsl.append((ddn[b_] * dl[b_]).sum() / (dl[b_] ** 2).sum())
    st.update(dZW_phi_mean=float(ddphi.mean()), dZW_phi_ci=ci(WbW @ ddphi / nW), natural_slope_dZW_per_dlogmA=slope, natural_slope_ci=ci(bsl),
              natural_pred_dZW_for_phi=float(slope * dlm.mean()), natural_pairs_r=float(np.corrcoef(ddn, dl)[0, 1]))

    def eff(L):
        f = flow(L) - fb; g = dd(L)
        return dict(flow=float(f.mean()), flow_ci=ci(WbW @ f / nW), ratio_to_phi=float(f.mean() / fphi.mean()),
                    ratio_ci=ci((WbW @ f) / np.maximum(WbW @ fphi, 1e-9)), mcnemar_W_calls=mcnemar(Lb.argmax(1) == iW, L.argmax(1) == iW),
                    dZW=float(g.mean()), dZW_ci=ci(WbW @ g / nW), dZW_ratio_to_phi=float(g.mean() / ddphi.mean()))
    # (b) single-feature dose, toward higher m_A
    sdf = float(acts[fs][:, fi].std()); sgn = float(np.sign(beta[fs][fi])); dose = {}
    for k in (0.16, 1.0, 2.0, 4.0, 8.0):
        dvec = np.zeros((nW, D[fs].D), np.float32); dvec[:, fi] = sgn * k * sdf; dose[str(k)] = eff(steer_multi(JW, fs, dvec))
    st['single_feature_dose_sd'] = dose; st['single_feature_sd'] = sdf; st['single_feature_beta'] = float(beta[fs][fi])
    log('dose', {k: (round(v['flow'], 4), round(v['dZW'], 4)) for k, v in dose.items()}, 'natural slope', slope, 'phi dZW', float(ddphi.mean())); res['steering'] = st; dump()
    # (c) joint steering of a whole site by beta_i * dlog m_A(phi), and with only the BH-significant features; permuted-beta control
    joint = {}; rngq = np.random.default_rng(SEED + 2)
    for s in CS:
        live = acts[s].std(0) > 0; b = np.where(live, beta[s], 0.0).astype(np.float32); bs = np.where(sig[s], b, 0.0).astype(np.float32)
        r = {}
        for nm, bv, sc in (('all_live_x1', b, 1.0), ('all_live_x3', b, 3.0), ('all_live_x10', b, 10.0), ('bh_sig_x1', bs, 1.0)):
            r[nm] = eff(steer_multi(JW, s, sc * dlm[:, None].astype(np.float32) * bv[None]))
        # residual-norm of the steering vector per unit dlog m_A, and the permuted-beta control (same feature values, shuffled across live features)
        r['steer_norm_per_unit'] = float((D[s].s_out * (t.from_numpy(b).to(dev) @ D[s].W_dec)).norm())
        li = np.where(live)[0]; pf = []
        for _ in range(NPERM):
            bp = b.copy(); bp[li] = b[rngq.permutation(li)]; pf.append(eff(steer_multi(JW, s, dlm[:, None].astype(np.float32) * bp[None]))['flow'])
        pf = np.array(pf); r['perm_control'] = dict(flows=pf.tolist(), mean=float(pf.mean()), q=ci(pf),
                                                     p_abs=float((1 + (np.abs(pf) >= abs(r['all_live_x1']['flow'])).sum()) / (NPERM + 1)))
        r['n_live'] = int(live.sum()); r['n_bh_sig'] = int(sig[s].sum()); joint[s] = r
        log(f'joint {s}: dZW x1 {r["all_live_x1"]["dZW"]:.4f} x10 {r["all_live_x10"]["dZW"]:.4f} (phi {ddphi.mean():.4f}, natural pred {slope * dlm.mean():.4f}); flow x1 {r["all_live_x1"]["flow"]:.4f} x3 {r["all_live_x3"]["flow"]:.4f} bh {r["bh_sig_x1"]["flow"]:.4f} perm mean {pf.mean():.4f}; phi {fphi.mean():.4f}')
    st['joint'] = joint; res['steering'] = st; res['seconds'] = time.time() - T0; dump(); log('done')


if __name__ == '__main__':
    main()
