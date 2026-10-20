"""E23x verify (skeptic): fresh test sample, reproduce the S31 + S58 / g freeze shares of the odd shift response, and add
the controls the E23x run lacked.

Model: official JEDI-linear 3-feature n64, exact forward (interp.jedi_core); S freezes through the exact head of
interp.e23x_centre_jedi (checked again here).  Sample: sample(1500, SEED) with SEED = 4313 (not used by E23x).
Edit: the E23x rigid translation, |c| 30 and 60 mrad x 8 azimuths (8 mirror pairs); odd O = (f(c) - f(-c)) / 2 per
standardised unit (19 live S, live h4, live h5, 16 z, W - Z margin).  Share removed = 1 - P(O_frozen) / P(O).

V1  reproduction: S31 + S58, S31, S58, all-g freezes; path share P(O - O_frz) / P(O); calls at 60 mrad.
V2  g freeze with S31 and S58 left free (S31, S58 set to their shifted values, all other S units computed with g held at
    the unshifted g): share removed by the rest of the g route.  If the g contribution passes through S31 + S58, ~0.
V3  matched random-subspace control (missing in E23x).  In the standardised 19-dim live-S space, a random k-dim subspace V
    is frozen: S_f = S_c - sd * V V^T ((S_c - S0) / sd) on the live units.  V = [e31, e58] reproduces arm 1 exactly
    (checked).  Draws are tilted toward the odd-response principal directions (V = orth(Sigma^(gamma/2) G), gamma ~ U(0, 4))
    so that they reach the S-level share of S31 + S58.  Families: 'all19' (k = 2, any live direction) and 'others17' (k in
    2..8, only the 17 live S units other than S31, S58).  Matched = S-level share within +-0.05 of S31 + S58's.
    H0: S31 + S58 remove the same downstream share as matched subspaces; reported: band, rank p, and the share predicted
    at S31 + S58's S-level share by a linear fit of downstream share on S-level share across all draws.
V4  centre edits on 3,000 random jets of this sample, |c| = 0.06 and 0.03 rings:
      soft_cmp / hard_cmp: E23x 5a / 5b (move a set, dilate the complement to hold M2; 5b dilates the soft particles);
      soft_self / hard_self: move the set by c and rescale only that set about the stored axis so its own M2 is held
        (total M2 held, the other particles untouched);
      soft_free / hard_free: move the set by c, nothing else.
    Per edit: feasible fraction, odd RMS per stage at 60 mrad, count- and pT-centroid move / |c|, odd RMS of
    q_count = 1/2 dlog sum r^2 and of q_softset = 1/2 dlog sum_{rank >= 14} r^2; S31 + S58 freeze share.
    soft / hard ratio of the odd RMS per stage on jointly feasible jets (bootstrap CI).
V5  180 deg rotation: share removed by S58, S31, both, g (reproduction of arm 6).
  ./job.sh e23x-verify 'python -m interp.e23x_verify'   -> $AI_RESULTS/jedi/e23x_verify.json
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from interp.e23w_gains_jedi import LIVE_S, ci, clean, shift  # noqa: E402
from interp.e23x_centre_jedi import (Head, Units, freeze, odd_even, perjet, share, share_point, calls_summary,  # noqa: E402
                                     centre_edit, C, PAIRS, PRING, SOFT, HARD, rotate_fixed)

OUT = Path(os.environ.get('AI_RESULTS', ROOT / 'results')) / 'jedi'
NB = 2000; DOWN = ('h4', 'h5', 'z', 'margin')
T0 = time.time()


def log(*a): print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)


def sets(x):
    x64 = x.astype(np.float64); rl = x64[..., 0] > 0; pt = np.where(rl, x64[..., 0], 0)
    e = np.where(rl, x64[..., 1], 0); p = np.where(rl, x64[..., 2], 0); rank = np.argsort(np.argsort(-pt, 1), 1)
    return x64, rl, pt, e, p, rank


def set_edit(x, kind, ce, cp, mode):
    """kind soft / hard; mode 'self' (rescale the moved set to hold its own M2) or 'free' (translation only)"""
    _, rl, pt, e, p, rank = sets(x)
    mv = rl & (pt < SOFT) & (rank >= HARD) if kind == 'soft' else rl & (rank < HARD)
    e1 = e + ce * mv; p1 = p + cp * mv; ok = mv.sum(1) > 0
    if mode == 'self':
        M2a = np.where(mv, pt * (e * e + p * p), 0).sum(1); M2b = np.where(mv, pt * (e1 * e1 + p1 * p1), 0).sum(1)
        k2 = M2a / np.maximum(M2b, 1e-30); ok &= (k2 >= 0.25) & (k2 <= 4); k = np.sqrt(np.clip(k2, 0.25, 4))[:, None]
        e1 = np.where(mv, e1 * k, e1); p1 = np.where(mv, p1 * k, p1)
    xe = x.copy(); xe[..., 1] = np.where(rl, e1, x[..., 1]).astype(np.float32); xe[..., 2] = np.where(rl, p1, x[..., 2]).astype(np.float32)
    return xe, ok


def meas(x, xe):
    _, rl, pt, e, p, rank = sets(x); _, _, _, e2, p2, _ = sets(xe)
    n = rl.sum(1); M0 = pt.sum(1); de, dp = e2 - e, p2 - p
    cnt = np.stack([de.sum(1) / n, dp.sum(1) / n], -1); ptc = np.stack([(pt * de).sum(1) / M0, (pt * dp).sum(1) / M0], -1)
    r2a, r2b = e * e + p * p, e2 * e2 + p2 * p2; soft = rl & (rank >= HARD)
    qn = 0.5 * np.log(r2b.sum(1) / r2a.sum(1)); qp = 0.5 * np.log((pt * r2b).sum(1) / (pt * r2a).sum(1))
    qs = 0.5 * np.log(np.maximum(np.where(soft, r2b, 0).sum(1), 1e-30) / np.maximum(np.where(soft, r2a, 0).sum(1), 1e-30))
    return cnt, ptc, qn, qp, qs


def odd(v): return np.stack([0.5 * (v[i] - v[j]) for i, j in PAIRS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per', type=int, default=1500); ap.add_argument('--seed', type=int, default=4313)
    ap.add_argument('--n_all19', type=int, default=300); ap.add_argument('--n_oth', type=int, default=400); ap.add_argument('--n5', type=int, default=3000)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    from interp.jedi_core import Jedi, sample
    from interp.e23_tail_jedi import sample_idx
    OUT.mkdir(parents=True, exist_ok=True); JS = OUT / f'e23x_verify{"_" + a.tag if a.tag else ""}.json'
    R = dict(node='E23x-verify', seed=a.seed, per=a.per); save = lambda: JS.write_text(json.dumps(clean(R)))
    xt, yt, _ = sample(a.per, a.seed); xt = np.ascontiguousarray(xt, np.float32)
    R['overlap_with_2311'] = int(np.isin(sample_idx(a.per, a.seed), sample_idx(a.per, 2311)).sum())
    J = len(xt); R['N'] = J
    net = Jedi(64); head = Head(net)
    B0 = net.run(xt, keep=('g', 'S', 'logits'), batch=4000); S0, g0, call0 = B0['S'].astype(np.float32), B0['g'], B0['logits'].argmax(1)
    R['acc'] = float((call0 == yt).mean())
    h4, h5, z, lo = head(S0); R['head_max_abs_dlogit'] = float(np.abs(lo - B0['logits']).max())
    l4 = [int(c) for c in np.flatnonzero(h4.std(0) > 1e-9)]; l5 = [int(c) for c in np.flatnonzero(h5.std(0) > 1e-9)]
    UN = Units(head, LIVE_S, l4, l5); U0, _ = UN(S0); sd = U0.std(0); sd[sd < 1e-9] = 1.0; sl = UN.sl
    log(f'N {J} acc {R["acc"]:.4f} head max |dlogit| {R["head_max_abs_dlogit"]:.1e} overlap {R["overlap_with_2311"]} live h4 {len(l4)} h5 {len(l5)}')
    WB = np.random.default_rng(11).poisson(1.0, (NB, J)).astype(np.float32)
    c06 = [i for i, (r, _, _, _) in enumerate(C) if r == 0.06]

    # ---------------- V1 / V2
    Sc = np.zeros((len(C), J, 64), np.float32); Sg = np.zeros_like(Sc)
    for i, (r, k, ce, cp) in enumerate(C):
        xs = shift(xt, ce, cp); Sc[i] = net.run(xs, keep=('S',), batch=4000)['S']
        Sg[i] = net.run_jetwise(xs, {'g': (np.ones(64), g0)}, keep=('S',), batch=4000)['S']
    R['g_hold_S31_S58_max_abs_change'] = float(np.abs(Sg[..., [31, 58]] - S0[None][..., [31, 58]]).max())
    lo_full = net.run(shift(xt[:500], C[-1][2], C[-1][3]), keep=('logits',))['logits']
    R['head_vs_full_shift_max_abs_dlogit'] = float(np.abs(lo_full - head(Sc[-1, :500])[3]).max())
    Uc, call_c = UN(Sc); O, E = odd_even(Uc, U0, sd)
    R['odd_rms'] = {st: float(np.sqrt((O[..., s] ** 2).mean())) for st, s in sl.items()}
    Sg2 = Sg.copy(); Sg2[..., [31, 58]] = Sc[..., [31, 58]]
    variants = {'S31_S58': freeze(Sc, S0, [31, 58]), 'S31': freeze(Sc, S0, [31]), 'S58': freeze(Sc, S0, [58]), 'g_all': Sg, 'g_except_S31_S58': Sg2}
    R['freeze'] = {}; R['calls'] = {}
    for nm, Sf in variants.items():
        Uf, cf = UN(Sf); Of, _ = odd_even(Uf, U0, sd); a_, b_, c_ = perjet(Of, sl), perjet(O, sl), perjet(O - Of, sl)
        R['freeze'][nm] = {st: dict(removed=share(a_[st], b_[st], WB)[0], path=float(c_[st].sum() / b_[st].sum())) for st in sl}
        R['calls'][nm] = calls_summary(call_c, cf, call0, yt, c06)
        log(nm, ' | '.join(f'{st} {v["removed"]["share"]:+.3f} {np.round(v["removed"]["ci"], 3)} path {v["path"]:.3f}' for st, v in R['freeze'][nm].items()),
            f'| calls {R["calls"][nm]["changed_unfrozen"]:.4f}->{R["calls"][nm]["changed_frozen"]:.4f} WZ {R["calls"][nm]["W_to_Z_unfrozen"]:.3f}->{R["calls"][nm]["W_to_Z_frozen"]:.3f}')
    save(); del Sg, Sg2, variants

    # ---------------- V3 matched random subspaces
    sdS = sd[sl['S']].astype(np.float64); li = np.array(LIVE_S)
    OS = O[..., sl['S']].reshape(-1, len(LIVE_S)).astype(np.float64); Sig = OS.T @ OS; tot = np.trace(Sig)
    i31, i58 = LIVE_S.index(31), LIVE_S.index(58)
    dZ = ((Sc[..., li] - S0[None][..., li]) / sdS).astype(np.float32)          # (C, J, 19) standardised change

    def proj_freeze(V):
        P = (V @ V.T).astype(np.float32); Sf = Sc.copy(); Sf[..., li] = Sc[..., li] - (dZ @ P) * sdS.astype(np.float32)
        return Sf

    def evald(V):
        sS = float(np.trace(V.T @ Sig @ V) / tot); Of, _ = odd_even(UN(proj_freeze(V))[0], U0, sd)
        return sS, share_point(Of, O, sl)
    e2 = np.zeros((len(LIVE_S), 2)); e2[i31, 0] = 1; e2[i58, 1] = 1
    ref_sS, ref = evald(e2); R['V3_ref_axis'] = dict(S_share_captured=ref_sS, share=ref)
    log('V3 reference [e31, e58]', ref_sS, ref)
    w, U = np.linalg.eigh(Sig); w = np.clip(w, 0, None); rng = np.random.default_rng(97)
    oth_idx = [i for i in range(len(LIVE_S)) if i not in (i31, i58)]
    wo, Uo = np.linalg.eigh(Sig[np.ix_(oth_idx, oth_idx)]); wo = np.clip(wo, 0, None)
    draws = {'all19': [], 'others17': []}
    for fam, n in (('all19', a.n_all19), ('others17', a.n_oth)):
        for _ in range(n):
            gam = rng.uniform(0, 4)
            if fam == 'all19':
                k = 2; M = (U * (w / w.max()) ** (gam / 2)) @ U.T @ rng.standard_normal((len(LIVE_S), k))
                V, _ = np.linalg.qr(M)
            else:
                k = int(rng.choice([2, 3, 4, 6, 8])); M = (Uo * (wo / wo.max()) ** (gam / 2)) @ Uo.T @ rng.standard_normal((len(oth_idx), k))
                Vo, _ = np.linalg.qr(M); V = np.zeros((len(LIVE_S), k)); V[oth_idx] = Vo
            sS, sh = evald(V)
            draws[fam].append(dict(k=k, gamma=gam, S_captured=sS, overlap31_58=float((V[[i31, i58]] ** 2).sum()), share=sh))
        log(fam, 'done', len(draws[fam]))
    R['V3_draws'] = draws
    tgt = R['freeze']['S31_S58']['S']['removed']['share']; obs = {st: R['freeze']['S31_S58'][st]['removed']['share'] for st in sl}
    R['V3_summary'] = {}
    for fam, rows in draws.items():
        xs_ = np.array([r_['share']['S'] for r_ in rows]); out = dict(n=len(rows), S_share_range=[float(xs_.min()), float(xs_.max())])
        m = np.abs(xs_ - tgt) <= 0.05; out['n_matched'] = int(m.sum())
        for st in DOWN:
            v = np.array([r_['share'][st] for r_ in rows]); d = dict()
            if m.sum() >= 5:
                vm = v[m]; d.update(matched_mean=float(vm.mean()), matched_lo=float(np.percentile(vm, 2.5)), matched_hi=float(np.percentile(vm, 97.5)),
                                    matched_max=float(vm.max()), p_rank=float((1 + (vm >= obs[st]).sum()) / (len(vm) + 1)),
                                    p_rank_two=float(min(1, 2 * min((1 + (vm >= obs[st]).sum()) / (len(vm) + 1), (1 + (vm <= obs[st]).sum()) / (len(vm) + 1)))))
            A = np.stack([np.ones_like(xs_), xs_], 1); beta, *_ = np.linalg.lstsq(A, v, rcond=None); res = v - A @ beta; s_ = float(res.std(ddof=2))
            pred = float(beta[0] + beta[1] * tgt); d.update(fit_intercept=float(beta[0]), fit_slope=float(beta[1]), pred_at_target=pred, resid_sd=s_,
                                                          obs=obs[st], z=float((obs[st] - pred) / s_), p_two=float(math.erfc(abs(obs[st] - pred) / s_ / math.sqrt(2))))
            out[st] = d
        R['V3_summary'][fam] = out; log('V3', fam, json.dumps(clean(out)))
    save(); del dZ

    # ---------------- V5 rotation 180
    xr = rotate_fixed(xt, 'rot180'); Sr = net.run(xr, keep=('S',), batch=4000)['S'].astype(np.float32)
    Srg = net.run_jetwise(xr, {'g': (np.ones(64), g0)}, keep=('S',), batch=4000)['S'].astype(np.float32)
    D = ((UN(Sr)[0] - U0) / sd)[None]; R['rot180'] = {}
    for nm, Sf in (('S58', freeze(Sr, S0, [58])), ('S31', freeze(Sr, S0, [31])), ('S31_S58', freeze(Sr, S0, [31, 58])), ('g_all', Srg)):
        Df = ((UN(Sf)[0] - U0) / sd)[None]; a_, b_ = perjet(Df, sl), perjet(D, sl)
        R['rot180'][nm] = {st: share(a_[st], b_[st], WB)[0] for st in sl}
    log('rot180', json.dumps(clean({k: {st: round(v[st]['share'], 3) for st in v} for k, v in R['rot180'].items()}))); save()

    # ---------------- V4 centre edits
    i5 = np.sort(np.random.default_rng(23).choice(J, min(a.n5, J), replace=False)); xx = xt[i5]; S0s = S0[i5]; U0s = U0[i5]
    edits = ('rigid', 'soft_cmp', 'hard_cmp', 'soft_self', 'hard_self', 'soft_free', 'hard_free'); data = {}
    c06p = np.flatnonzero(PRING == 1)
    for ed in edits:
        Se = np.zeros((len(C), len(xx), 64), np.float32); ok = np.ones(len(xx), bool); M = [[] for _ in range(5)]
        for i, (r, k, ce, cp) in enumerate(C):
            if ed == 'rigid': xe = shift(xx, ce, cp)
            elif ed.endswith('cmp'): xe, o_ = centre_edit(xx, ed.split('_')[0], ce, cp); ok &= o_
            else: xe, o_ = set_edit(xx, ed.split('_')[0], ce, cp, ed.split('_')[1]); ok &= o_
            for lst, v in zip(M, meas(xx, xe)): lst.append(v)
            Se[i] = Sc[i, i5] if ed == 'rigid' else net.run(xe, keep=('S',), batch=4000)['S']
        cnt, ptc, qn, qp, qs = [np.stack(m_) for m_ in M]
        Oe, _ = odd_even(UN(Se)[0], U0s, sd); Of, _ = odd_even(UN(freeze(Se, S0s, [31, 58]))[0], U0s, sd)
        oc, op, oqn, oqp, oqs = odd(cnt)[c06p][:, ok], odd(ptc)[c06p][:, ok], odd(qn)[c06p][:, ok], odd(qp)[c06p][:, ok], odd(qs)[c06p][:, ok]
        m = dict(frac_feasible=float(ok.mean()), count_centroid_over_c=float(np.median(np.hypot(oc[..., 0], oc[..., 1]) / 0.06)),
                 pt_centroid_over_c=float(np.median(np.hypot(op[..., 0], op[..., 1]) / 0.06)), q_count_odd_rms_pct=float(100 * np.sqrt((oqn ** 2).mean())),
                 q_pt_odd_rms_pct=float(100 * np.sqrt((oqp ** 2).mean())), q_softset_odd_rms_pct=float(100 * np.sqrt((oqs ** 2).mean())))
        Wj = np.random.default_rng(37).poisson(1.0, (NB, int(ok.sum()))).astype(np.float32)
        a_, b_ = perjet(Of[c06p][:, ok], sl), perjet(Oe[c06p][:, ok], sl)
        m['freeze_S31_S58'] = {st: share(a_[st], b_[st], Wj)[0] for st in sl}
        m['odd_rms_06'] = {st: float(np.sqrt((Oe[c06p][:, ok][..., s] ** 2).mean())) for st, s in sl.items()}
        data[ed] = dict(O=Oe[c06p], ok=ok); R.setdefault('V4', {})[ed] = m
        log('V4', ed, json.dumps(clean({k_: v_ for k_, v_ in m.items() if k_ != 'freeze_S31_S58'})),
            'frz h4', round(m['freeze_S31_S58']['h4']['share'], 3), 'margin', round(m['freeze_S31_S58']['margin']['share'], 3))
        save()
    R['V4_ratio'] = {}
    for s_, h_ in (('soft_cmp', 'hard_cmp'), ('soft_self', 'hard_self'), ('soft_free', 'hard_free')):
        okb = data[s_]['ok'] & data[h_]['ok']; Wb = np.random.default_rng(31).poisson(1.0, (NB, int(okb.sum()))).astype(np.float32); out = dict(n=int(okb.sum()))
        for st, s in sl.items():
            pa = (data[s_]['O'][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2)); pb = (data[h_]['O'][:, okb][..., s].astype(np.float64) ** 2).sum((0, 2))
            rb = np.sqrt((Wb @ pa) / (Wb @ pb)); out[st] = dict(ratio_soft_over_hard=float(np.sqrt(pa.sum() / pb.sum())), ci=ci(rb))
        R['V4_ratio'][f'{s_}/{h_}'] = out; log('V4 ratio', s_, h_, json.dumps(clean(out)))
    R['wall_s'] = time.time() - T0; save(); log('done ->', JS)


if __name__ == '__main__':
    main()
