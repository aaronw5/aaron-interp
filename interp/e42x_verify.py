"""E42x-verify (verifier for E42x, claim A20t): fresh-sample rerun of the rigid W-pair edit, plus a control that does
not use the non-W edit (toward vs away).

Reason: on the E42x sample the official network's per-variant responses at T80 were dW = +0.03 (W edit alone) and
dN = +0.24 (control alone), so the negative T80 W - N comes from the control raising the score; the W edit alone sits
between BLIND (-0.04) and AWARE (+0.14).  A W-mass reading is tested here without the N edit.

PLAN (fixed before the run)
Sample: jedi_core.sample(PER, 4521), PER = 7000 per class (35,000 jets; seed new; rows shared with E42x 4421 and
  E42 / E42v counted, not removed).  Subjets, W pair, top window, eligibility, the Wr and ND edits and their acceptance
  are E42x's (functions imported from interp.e42x_top_wpair_matched_jedi).
(1) Replication: Wr - ND at T60 / T100 / T80 (official + the 7 E39 networks + an untrained redraw; BLIND / AWARE GBDT
  references, 2-fold cross-fit on the 35,000 natural jets), and the W edit alone vs the references:
  dW_net - dW_BLIND (no N control).
(2) Toward vs away (TA): the T80-eligible tops (top-window, reclustered m_W in 60-70 or 90-100 GeV) whose edited kt3
  W pair has |m_W0 - 80.4| > 8 GeV get two Wr edits of the same pair: toward (t = 80.4) and away (t = 2 m_W0 - 80.4,
  i.e. 60-70 -> 40-60 GeV and 90-100 -> 100-120 GeV), each holding m and tau32 (E42x's Wr solver).  Per jet
  C = s_toward - s_away (equal |delta m_W|, same pair, same constraints).  Acceptance for both: |dm|/m < 1 %,
  |dtau32|/tau32 < 1 %, and one reclustered pair mass within 2 GeV of t.
  Low side (m_W0 < 80.4): toward opens the pair, away closes it; high side: the reverse.  The balanced mean
  Cbal = (mean_low + mean_high) / 2 cancels a response that depends only on the opening direction (odd) and, to first
  order, one that depends only on the size of the deformation (even).
  Predictions: m_W reader: C > 0 on both sides (size from AWARE).  Deformation-only: Cbal = C_BLIND (both sides
  opposite in sign if odd, equal if even).  H0: Cbal_net = Cbal_BLIND; also Cbal_net = 0.  H1: Cbal_net > Cbal_BLIND.
  Stratified bootstrap (2,000, resampling within side), sign-flip permutation (10,000, two-sided).  Regression of C on
  the toward - away differences of tau21, mSD, D and a side code (+-0.5): intercept = Cbal at zero side effects.
Holm over the 8 trained networks within each family (TA net - BLIND; TA vs 0; W - N per set over 8 x 3; dW - BLIND 8 x 3).
Outputs: $AI_RESULTS/jedi/e42x_verify_edit.json, $AI_RESULTS/jedi/e42x_verify.json.
Run: AI_PY=conda ./job.sh e42x-verify 'python -m interp.e42x_verify --part edit &&
     /j-jepa-vol/phat-jet-aaron/env/bin/python -m interp.e42x_verify --part jedi'
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

os.environ.setdefault('KERAS_BACKEND', 'jax')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp import e42x_top_wpair_matched_jedi as X  # noqa: E402

T00 = time.time()
PER = int(os.environ.get('E42XV_PER', '7000'))
SEED = 4521
OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
TAG = 'e42x_verify' if PER >= 1000 else 'e42x_verify_smoke'
TMP = Path('/tmp/e42xv')
TARGET = {'T60': 60.0, 'T100': 100.0, 'T80': 80.4}
TOLW = {'T60': 5.0, 'T100': 5.0, 'T80': 2.0}
CONTR = ('T60', 'T100', 'T80', 'TA')
NB, NPERM = 2000, 10000


def log(*a):
    print(f'[{time.time() - T00:7.0f}s]', *a, flush=True)


def edit_one(P, lab, keys):
    from interp import h4_top_wmass as H
    jt = H.Jet(P, lab); fj = X.Fast(jt); tau0 = jt.tau32(np.zeros((3, 2)))
    pm0, m0 = jt.pair_masses(); w = min(pm0, key=lambda p: abs(pm0[p] - 80.4)); a, b = w; c = 3 - a - b; mW0 = pm0[w]
    out = dict(mW0=mW0)
    for key in keys:
        if key == 'AWAY':
            if abs(mW0 - 80.4) <= 8: continue
            t = 2 * mW0 - 80.4
        else:
            t = TARGET[key]
        R = {}
        W = X.w_rigid(jt, fj, t, tau0, m0, a, b, c)
        if W is not None:
            DW = X.disp(jt, W['s']); R['Wr'] = dict(P=jt.build(W['s'])[0].astype(np.float32), f=W['f'], th=W['th'], D=DW, t=t)
            if key != 'AWAY':
                N = X.n_matched(jt, fj, a, b, c, W['f'], m0, DW)
                if N is not None: R['ND'] = dict(P=jt.build(N['s'])[0].astype(np.float32), f=N['fac'], D=X.disp(jt, N['s']), t=t)
        out[key] = R
    return out


def part_edit():
    from interp import h4_top_wmass as H
    from interp.h29_jedi_wpair_rebuild import p4_of, axis, model_x
    from interp.jedi_core import sample
    from interp.e42_verify import sample_idx
    TMP.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    D, meta = {}, dict(node='E42x-verify', per_class=PER, seed=SEED, contrasts={})
    x, y, _ = sample(PER, SEED); idx = sample_idx(PER, SEED)
    meta['overlap_rows'] = dict(E42x=int(np.isin(idx, sample_idx(5000, 4421)).sum()),
                                E42_E42v=int(np.isin(idx, np.concatenate([sample_idx(2000, 4201), sample_idx(2000, 4202), sample_idx(3000, 4302)])).sum()))
    real = x[..., 0] > 0; nn = real.sum(1).astype(np.int64)
    assert all(real[j, :nn[j]].all() for j in range(len(nn))), 'padding not contiguous'
    P4 = H._identity(p4_of(x[..., 0], x[..., 1], x[..., 2]), nn).astype(np.float32)
    meta['identity_rebuild_max_abs'] = float(np.abs(model_x(P4, nn) - x).max()); D['U_st'] = x; del x
    log(f'sample {SEED}: N {len(y)}; overlap {meta["overlap_rows"]}; identity rebuild {meta["identity_rebuild_max_abs"]:.2e}')
    D['y'] = y
    O = X.allobs(P4, nn); log('observables on all jets')
    top = y == 4; win = top & (O['mSD'] > 105) & (O['mSD'] < 210) & (nn >= 3)
    eT = win & (O['mW'] > 65) & (O['mW'] < 95); e80 = win & (((O['mW'] >= 60) & (O['mW'] < 70)) | ((O['mW'] > 90) & (O['mW'] <= 100)))
    meta['eligible'] = dict(tops=int(top.sum()), window=int(win.sum()), T60_T100=int(eT.sum()), T80=int(e80.sum())); log(meta['eligible'])
    rowsU = np.flatnonzero(eT | e80); F = np.zeros((len(rowsU), P4.shape[1], 4)); F[:] = P4[rowsU]; LAB = H.kt3_batch(F, nn[rowsU]); del F
    res = {}; t1 = time.time()
    for ii, j in enumerate(rowsU):
        keys = (['T60', 'T100'] if eT[j] else []) + (['T80', 'AWAY'] if e80[j] else [])
        try:
            res[j] = edit_one(P4[j, :nn[j]], LAB[ii, :nn[j]], keys)
        except Exception as ex:  # noqa: BLE001
            res[j] = dict(error=repr(ex))
        if (ii + 1) % 500 == 0: log(f'  rigid edits {ii + 1}/{len(rowsU)} ({time.time() - t1:.0f}s)')
    meta['rigid_edit_errors'] = sum('error' in r for r in res.values()); log(f'rigid edits done ({time.time() - t1:.0f}s), errors {meta["rigid_edit_errors"]}')
    VAR = {}
    for key in ('T60', 'T100', 'T80', 'AWAY'):
        rows = np.flatnonzero(e80 if key in ('T80', 'AWAY') else eT)
        for v in ('Wr', 'ND'):
            rr = [j for j in rows if 'error' not in res[j] and key in res[j] and v in res[j][key]]
            if not rr: continue
            Pv = np.zeros((len(rr), P4.shape[1], 4), np.float32)
            for i, j in enumerate(rr): Pv[i, :nn[j]] = res[j][key][v]['P']
            rr = np.array(rr, np.int64); Ov = X.allobs(Pv, nn[rr]); t = np.array([res[j][key][v]['t'] for j in rr])
            dm = np.abs(Ov['m'] / O['m'][rr] - 1); dtau = np.abs(Ov['tau32'] / O['tau32'][rr] - 1)
            pm = np.column_stack([Ov['mWc'], Ov['mo1'], Ov['mo2']]); near = np.abs(pm - t[:, None]).min(1) < 2.0
            if v == 'Wr':
                acc_e42x = (dm < 0.01) & (dtau < 0.01) & (np.abs(Ov['mW'] - t) < TOLW.get(key, 2.0))
                acc_ta = (dm < 0.01) & (dtau < 0.01) & near
            else:
                acc_e42x = (dm < 0.01) & (np.abs(Ov['mW'] - O['mW'][rr]) < 2); acc_ta = acc_e42x
            VAR[(key, v)] = dict(rows=rr, P=Pv, O=Ov, acc=acc_e42x, acc_ta=acc_ta,
                                 mW0=np.array([res[j]['mW0'] for j in rr]), f=np.array([res[j][key][v]['f'] for j in rr]))
            meta.setdefault('counts', {})[f'{key}_{v}'] = dict(solved=int(len(rr)), accepted=int(acc_e42x.sum()), accepted_ta=int(acc_ta.sum()))
    log(f'counts {meta["counts"]}')
    spec = {'T60': (('T60', 'Wr', 'acc'), ('T60', 'ND', 'acc')), 'T100': (('T100', 'Wr', 'acc'), ('T100', 'ND', 'acc')),
            'T80': (('T80', 'Wr', 'acc'), ('T80', 'ND', 'acc')), 'TA': (('T80', 'Wr', 'acc_ta'), ('AWAY', 'Wr', 'acc_ta'))}
    for cn, ((kW, vW, aW), (kN, vN, aN)) in spec.items():
        if (kW, vW) not in VAR or (kN, vN) not in VAR: meta['contrasts'][cn] = dict(n=0); continue
        A, B = VAR[(kW, vW)], VAR[(kN, vN)]; ar = np.intersect1d(A['rows'][A[aW]], B['rows'][B[aN]])
        meta['contrasts'][cn] = dict(n=int(len(ar)))
        if len(ar) < 5: continue
        D[f'rows__{cn}'] = ar; n0 = nn[ar]; Xid = model_x(P4[ar], n0, axis(P4[ar], n0)); D[f'X__{cn}__id'] = Xid
        for k in X.OBSK: D[f'O__{cn}__id__{k}'] = O[k][ar]
        se = {}
        for tag, V in (('W', A), ('N', B)):
            ii = np.searchsorted(V['rows'], ar); Pv = V['P'][ii]; Xv = model_x(Pv, n0, axis(Pv, n0)); D[f'X__{cn}__{tag}'] = Xv
            for k in X.OBSK: D[f'O__{cn}__{tag}__{k}'] = V['O'][k][ii]
            ptw = Xid[..., 0].astype(np.float64)
            D[f'O__{cn}__{tag}__D'] = (ptw * np.hypot(Xv[..., 1] - Xid[..., 1], Xv[..., 2] - Xid[..., 2])).sum(1) / ptw.sum(1)
            D[f'P__{cn}__{tag}__f'] = V['f'][ii]
            se[tag] = {k: float((V['O'][k][ii] - O[k][ar]).mean()) for k in ('m', 'mSD', 'tau21', 'tau32', 'mW', 'mWc', 'w1', 'w2', 'w3')}
            se[tag]['D'] = float(D[f'O__{cn}__{tag}__D'].mean()); se[tag]['f_median'] = float(np.median(V['f'][ii]))
        D[f'mW0__{cn}'] = A['mW0'][np.searchsorted(A['rows'], ar)]
        se['W_minus_N'] = {k: float((D[f'O__{cn}__W__{k}'] - D[f'O__{cn}__N__{k}']).mean()) for k in ('tau21', 'tau32', 'mSD', 'D', 'mWc', 'w1')}
        if cn == 'TA': se['n_low'] = int((D['mW0__TA'] < 80.4).sum()); se['n_high'] = int((D['mW0__TA'] > 80.4).sum())
        meta['contrasts'][cn]['side_effects'] = se; log(f'{cn}: n {len(ar)} {se}')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from interp.h4_top_wmass import _auc
    fold = np.random.default_rng(4522).permutation(len(y)) % 2; yb = (y == 4).astype(int); meta['ref'] = {}
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    for nm, fs in (('BLIND', X.FBLIND), ('AWARE', X.FAWARE)):
        Xn = np.column_stack([O[k] for k in fs]); mods = []; oof = np.zeros(len(y))
        for fo in (0, 1):
            m_ = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Xn[fold != fo], yb[fold != fo]); mods.append(m_)
            oof[fold == fo] = m_.predict_proba(Xn[fold == fo])[:, 1]
        meta['ref'][nm] = dict(features=list(fs), auc_oof=float(_auc(oof, yb)))
        for kk in [k for k in D if k.startswith('X__')]:
            _, cn, v = kk.split('__'); rr = D[f'rows__{cn}']; Xv = np.column_stack([D[f'O__{cn}__{v}__{k}'] for k in fs]); p = np.zeros(len(rr))
            for fo in (0, 1):
                sel = fold[rr] == fo
                if sel.any(): p[sel] = mods[1 - fo].predict_proba(Xv[sel])[:, 1]
            D[f'{nm}__{cn}__{v}'] = lo(p)
        log(f'reference {nm}: {meta["ref"][nm]}')
    np.savez_compressed(TMP / 'inputs.npz', **D)
    (OUT / f'{TAG}_edit.json').write_text(json.dumps(X.clean(meta), indent=1)); log('edit part written')


# ------------------------------------------------------------------------------------------------ statistics helpers
def boot_mean_ci(x, seed, side=None):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); ms = []
    if side is None:
        for _ in range(NB): ms.append(x[rng.integers(0, len(x), len(x))].mean())
    else:
        lo_, hi_ = x[side == 0], x[side == 1]
        for _ in range(NB): ms.append(0.5 * (lo_[rng.integers(0, len(lo_), len(lo_))].mean() + hi_[rng.integers(0, len(hi_), len(hi_))].mean()))
    return [float(np.quantile(ms, .025)), float(np.quantile(ms, .975))]


def stat(x, side=None):
    return float(x.mean()) if side is None else float(0.5 * (x[side == 0].mean() + x[side == 1].mean()))


def perm(x, rng, side=None):
    x = np.asarray(x, float); obs = abs(stat(x, side)); cnt = 0; done = 0
    wt = np.full(len(x), 1.0 / len(x)) if side is None else np.where(side == 0, 0.5 / max((side == 0).sum(), 1), 0.5 / max((side == 1).sum(), 1))
    while done < NPERM:
        k = min(1000, NPERM - done); s = rng.choice([-1.0, 1.0], (k, len(x))); cnt += int(np.sum(np.abs((s * x) @ wt) >= obs - 1e-15)); done += k
    return float((cnt + 1) / (NPERM + 1))


def summ(x, seed, rng, side=None):
    return dict(mean=stat(x, side), ci=boot_mean_ci(x, seed, side), p=perm(x, rng, side))


def part_jedi():
    from interp.jedi_core import Jedi
    from interp.e42_top_wpair_jedi import score, call, mcnemar, holm
    rng = np.random.default_rng(4523)
    Z = np.load(TMP / 'inputs.npz'); meta = json.loads((OUT / f'{TAG}_edit.json').read_text())
    R = dict(node='E42x-verify', plan='module docstring (fixed before the run)', edit=meta, refs={}, nets={})
    Xk = [k for k in Z.files if k.startswith('X__')]
    side = (Z['mW0__TA'] > 80.4).astype(int) if 'mW0__TA' in Z.files else None

    def analyse(sc, lg=None):
        out = {}
        for cn in CONTR:
            if f'{cn}__id' not in sc: continue
            s0, sW, sN = sc[f'{cn}__id'], sc[f'{cn}__W'], sc[f'{cn}__N']; dW, dN = sW - s0, sN - s0; x = dW - dN
            bW = Z[f'BLIND__{cn}__W'] - Z[f'BLIND__{cn}__id']; bN = Z[f'BLIND__{cn}__N'] - Z[f'BLIND__{cn}__id']
            sd = side if cn == 'TA' else None; sdd = 4530 + len(x)
            M = dict(n=int(len(x)), dW=summ(dW, sdd, rng, sd), dN=summ(dN, sdd, rng, sd), W_minus_N=summ(x, sdd, rng, sd),
                     W_minus_N_minus_BLIND=summ(x - (bW - bN), sdd, rng, sd), dW_minus_BLIND=summ(dW - bW, sdd, rng, sd),
                     dN_minus_BLIND=summ(dN - bN, sdd, rng, sd))
            if cn == 'TA':
                for nm_, s_ in (('low', 0), ('high', 1)):
                    m_ = side == s_; M[f'side_{nm_}'] = dict(n=int(m_.sum()), C=summ(x[m_], sdd + s_, rng), C_minus_BLIND=summ((x - (bW - bN))[m_], sdd + s_, rng))
                cov = [Z[f'O__TA__W__{k}'] - Z[f'O__TA__N__{k}'] for k in ('tau21', 'mSD', 'D')]
                C_ = np.column_stack(cov); s_ = C_.std(0); s_[s_ < 1e-12] = 1
                Am = np.column_stack([np.ones(len(x)), C_ / s_, side - 0.5]); beta = np.linalg.lstsq(Am, x, rcond=None)[0]; bi = np.random.default_rng(4527); ints = []
                for _ in range(NB):
                    ii = bi.integers(0, len(x), len(x)); ints.append(np.linalg.lstsq(Am[ii], x[ii], rcond=None)[0][0])
                M['regression'] = dict(covariates=['tau21', 'mSD', 'D', 'side'], intercept=float(beta[0]), intercept_ci=[float(np.quantile(ints, .025)), float(np.quantile(ints, .975))])
            if lg is not None:
                M['calls'] = dict(changed_W=int((call(lg[f'{cn}__W']) != call(lg[f'{cn}__id'])).sum()), changed_N=int((call(lg[f'{cn}__N']) != call(lg[f'{cn}__id'])).sum()),
                                  mcnemar_W_vs_N=mcnemar(call(lg[f'{cn}__W']), call(lg[f'{cn}__N'])))
            out[cn] = M
        return out

    for rf in ('BLIND', 'AWARE'):
        sc = {k[len(rf) + 2:]: Z[k] for k in Z.files if k.startswith(f'{rf}__')}
        R['refs'][rf] = analyse(sc)
        log(f'{rf}: ' + '; '.join(f'{c} dW {M["dW"]["mean"]:+.3f} dN {M["dN"]["mean"]:+.3f} W-N {M["W_minus_N"]["mean"]:+.3f} {np.round(M["W_minus_N"]["ci"], 3)}' for c, M in R['refs'][rf].items()))
    xs, ys = np.asarray(Z['U_st'], np.float32), Z['y']   # stored inputs (no second sample() read in this process: memory)
    nets = [('official_n64', None)] + [(n, n) for n in X.EDIT_NETS] + [('untrained_s0', 'untrained')]
    for name, kind in nets:
        t1 = time.time()
        try:
            if kind is None: M_ = Jedi(64, '3-feature')
            elif kind == 'untrained': M_ = Jedi(64, '3-feature', untrained=True, seed=0)
            else:
                from interp.e33_program_recurrence_jedi import Model
                M_ = Model(kind)
            run = lambda x, M_=M_: M_.run(x)['logits']
        except Exception as e:  # noqa: BLE001
            log(f'{name}: load failed {e!r}'); R['nets'][name] = dict(error=repr(e)); continue
        acc = float((run(xs).argmax(1) == ys).mean())
        sc, lg = {}, {}
        for k in Xk: L = run(np.asarray(Z[k], np.float32)); lg[k[3:]] = L; sc[k[3:]] = score(L)
        R['nets'][name] = dict(acc_stored=acc, contrasts=analyse(sc, lg))
        C = R['nets'][name]['contrasts']
        log(f'{name}: acc {acc:.4f} ({time.time() - t1:.0f}s)')
        for c, M in C.items():
            log(f'  {c} n {M["n"]}: dW {M["dW"]["mean"]:+.3f} {np.round(M["dW"]["ci"], 3)} dN {M["dN"]["mean"]:+.3f} {np.round(M["dN"]["ci"], 3)} '
                f'W-N {M["W_minus_N"]["mean"]:+.3f} {np.round(M["W_minus_N"]["ci"], 3)} p {M["W_minus_N"]["p"]:.3g}; W-N-BLIND {M["W_minus_N_minus_BLIND"]["mean"]:+.3f} '
                f'{np.round(M["W_minus_N_minus_BLIND"]["ci"], 3)} p {M["W_minus_N_minus_BLIND"]["p"]:.3g}; dW-BLIND {M["dW_minus_BLIND"]["mean"]:+.3f} {np.round(M["dW_minus_BLIND"]["ci"], 3)} p {M["dW_minus_BLIND"]["p"]:.3g}')
            if c == 'TA': log(f'    TA low {M["side_low"]}; high {M["side_high"]}; regression {M["regression"]}')
        del M_
    trained = [n for n, _ in nets if n != 'untrained_s0' and 'error' not in R['nets'][n]]
    for st in ('W_minus_N', 'W_minus_N_minus_BLIND', 'dW_minus_BLIND'):
        for fam_name, keys in (('TA', ('TA',)), ('sets', ('T60', 'T100', 'T80'))):
            fam = [(n, k) for n in trained for k in keys if k in R['nets'][n]['contrasts']]
            hp = holm([R['nets'][n]['contrasts'][k][st]['p'] for n, k in fam])
            for (n, k), h in zip(fam, hp): R['nets'][n]['contrasts'][k][st]['p_holm'] = float(h)
            R[f'holm_{st}_{fam_name}_family_size'] = len(fam)
    (OUT / f'{TAG}.json').write_text(json.dumps(X.clean(R), indent=1)); log(f'wrote {OUT}/{TAG}.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['edit', 'jedi'])
    a = ap.parse_args(); rc = 0
    try:
        part_edit() if a.part == 'edit' else part_jedi()
    except BaseException:  # noqa: BLE001
        import traceback; traceback.print_exc(); rc = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)
