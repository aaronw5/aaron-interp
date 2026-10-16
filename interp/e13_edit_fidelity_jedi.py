"""E13 (A1, B): does the float+mask program (E3m) predict the fixed-point network's response to particle-level edits?

Sample: sample(per_class=2000, seed=3) = 10,000 test jets.  Every edit changes one particle per jet (or inserts one) and
keeps the pT rank order; jets where it would not are skipped (N per edit reported).

Edits (before -> after):
  flip_r{r}    particle at rank r in {0,2,13,14,30} reflected through the jet axis (Δη,Δφ) -> (-Δη,-Δφ); |r| > 0.01 rad.
  rot_r{r}     matched control: same particle rotated about the axis by the largest angle that keeps it on the same side
               of all 9 live angular cut lines (a29 a6 a36 a40 a43 a46 a3 a57 a30), stopping a margin
               max(0.1·arc, 0.004 rad / |r|) before the arc edge (input LSB in Δη, Δφ is 0.0014 rad).
  a32_x        particle with 9-13 GeV: set to 10.0 GeV (before) and 12.0 GeV (after); a32 edge 11.0 GeV
  a32_c_lo     control 5 -> 7 GeV (particle with 4-8 GeV);  a32_c_hi  control 13 -> 15 GeV (particle with 12-16 GeV)
  a56_x        30 -> 32 GeV (particle with 29-33 GeV), a56 edge 31.0 GeV; a56_c_lo 25 -> 27; a56_c_hi 33 -> 35
  split        hardest particle split into two collinear halves (count +1, ΣpT unchanged), reinserted at their rank
  add1         one 1 GeV particle added at ΔR = 0.1 (azimuth uniform, seed 3), inserted at its rank
  (split/add1 skip jets with 64 real particles)
Per edit: network (fixed point, jedi_core) vs program (float + HGQ zero-bit masks, as interp/e3m_masks.py 'mask'):
  dP per live P unit: network vs program (= identity b·d|A| + w·dΣ_{A∩window} x, float) and vs the no-window identity;
  dS per live S unit; dlogits per class; call flips (rate, agreement on whether, agreement on the new call).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(ROOT / 'logs' / 'e13.log', 'w')
def log(*a):
    s = ' '.join(str(v) for v in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
t0 = time.time()

ANG = [29, 6, 36, 40, 43, 46, 3, 57, 30]
e1 = json.loads((OUT / 'e1_e3.json').read_text())['E1']
LIVE_P = [int(c) for c in np.argsort(-np.array(e1['P']['changed'])) if e1['P']['changed'][c] > 0.001]
LIVE_S = [int(c) for c in np.argsort(-np.array(e1['S']['changed'])) if e1['S']['changed'][c] > 0.001]
log('live P units (E1 >0.1 % calls):', LIVE_P); log('live S units (E1 >0.1 % calls):', LIVE_S)

net = Jedi(64); L = net.L
KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}
MASK = {name: jnp.asarray(np.asarray(L[name].iq.quantizer.b) > 0, jnp.float32)
        for name in ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')}
qsM = np.asarray(MASK['qs'])[0]                  # (64 slots, 64 channels) pooling window of P
ql1 = L['l1'].iq.quantizer
fbits = (np.asarray(ql1.b) - np.asarray(ql1.i) - np.asarray(ql1.k))[0]   # (64 slots, 3) fractional bits of the input
log('l1 input fractional bits (pT, Δη, Δφ) at slots 0,2,5,10,13,14,20,30,40,63:',
    {s: fbits[s].astype(int).tolist() for s in (0, 2, 5, 10, 13, 14, 20, 30, 40, 63)})
qq = L['qs'].iq.quantizer
fq = (np.asarray(qq.b) - np.asarray(qq.i) - np.asarray(qq.k))[0]           # (slots, channels) fractional bits of the pool input
log('pool quantizer (qs) bits b / fractional bits f for the angular channels at slots 0, 2, 13, 14, 30:')
for c in ANG + [32, 56]:
    log(f'  ch{c}: ' + ', '.join(f's{s_}: b={int(np.asarray(qq.b)[0, s_, c])} f={int(fq[s_, c])}' for s_ in (0, 2, 13, 14, 30)))
log('P window (slots kept by the qs mask) for live P units:',
    {c: f'{int(qsM[:, c].sum())}/64, missing {np.flatnonzero(qsM[:, c] == 0).tolist()[:12]}' for c in LIVE_P})


def mk(fixed):
    """fixed = set of layers whose fixed-point quantizer is kept; every other layer is float x its zero-bit mask"""
    def q(name, t):
        if name in fixed: return L[name].iq(t, training=False)
        return t * MASK[name] if name in MASK else t
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z
    def pool(k, a):
        return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    def f(xn):
        a = dense('l1', xn); P = pool('qs', a); g = dense('dd', P); s = dense('l2', a)
        if 'add' in fixed:
            qs_, qd_ = L['add'].iq([s, g], training=False); t = qs_ + qd_
        else: t = s + g
        u = dense('l3', t); S = pool('qs1', u); h = dense('d5', dense('d4', S))
        lo = dense('d7', jax.nn.relu(dense('d6', h, act=False)))
        J = xn.shape[0]
        return dict(P=P.reshape(J, -1), S=S.reshape(J, -1), logits=lo.reshape(J, -1))
    return jax.jit(f)
GROUPS = dict(r1=('l1', 'qs'), r2=('l2', 'dd', 'add', 'l3', 'qs1'), head=('d4', 'd5', 'd6', 'd7'))
ALLQ = sum(GROUPS.values(), ())
FN = dict(net=mk(set(ALLQ)), prog=mk(set()), **{f'h_{g}': mk(set(v)) for g, v in GROUPS.items()})
HYB = [f'h_{g}' for g in GROUPS]     # program + one group's fixed-point quantizers

K1, b1 = net.KB['l1']; scP = net.pool_scale[0]
alpha = K1 / net.scale[:, None]; beta = b1 - (K1 * (net.shift / net.scale)[:, None]).sum(0)   # physical cut coefficients


def run(x, which):
    out = {k: [] for k in ('P', 'S', 'logits')}
    for i in range(0, len(x), 5000):
        r = FN[which](jnp.asarray(net.inputs(x[i:i + 5000], 64)))
        for k in out: out[k].append(np.asarray(r[k], np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def ident(x, window):
    """b·|A| + w·Σ_A x per channel (float64), A = passing particles (∩ the qs window if window), times the pool scale"""
    pre = net.inputs(x, 64).astype(np.float64) @ K1 + b1           # (J, 64 slots, 64 ch)
    v = np.maximum(pre, 0)
    if window: v = v * qsM[None]
    return v.sum(1) * scP


# ---------------------------------------------------------------- reference: unedited jets
x0, y0, _ = sample(per_class=2000, seed=0)
c0n = run(x0, 'net')['logits'].argmax(1); c0p = run(x0, 'prog')['logits'].argmax(1)
for h in HYB:
    ch = run(x0, h)['logits'].argmax(1)
    log(f'reference seed 0: {h} (program + fixed-point quantizers of {GROUPS[h[2:]]}) acc {(ch == y0).mean()*100:.2f} %, '
        f'calls differ from network {(ch != c0n).mean()*100:.2f} %')
c0j = net.run(x0)['logits'].argmax(1)
log(f'reference seed 0 (N={len(x0)}): network acc {(c0n == y0).mean()*100:.2f} % (jedi_core {(c0j == y0).mean()*100:.2f} %, '
    f'same calls {(c0n == c0j).mean()*100:.2f} %), program acc {(c0p == y0).mean()*100:.2f} %, calls differ {(c0n != c0p).mean()*100:.2f} %')
x, y, _ = sample(per_class=2000, seed=3)
real = x[..., 0] > 0; nreal = real.sum(1)
B = dict(net=run(x, 'net'), prog=run(x, 'prog'))
cbn, cbp = B['net']['logits'].argmax(1), B['prog']['logits'].argmax(1)
idw = ident(x, True)
res = dict(N=len(x), seed=3, live_P=LIVE_P, live_S=LIVE_S, ref_seed0=dict(acc_net=float((c0n == y0).mean()),
           acc_prog=float((c0p == y0).mean()), calls_differ=float((c0n != c0p).mean())),
           unedited=dict(acc_net=float((cbn == y).mean()), acc_prog=float((cbp == y).mean()),
                         calls_differ=float((cbn != cbp).mean()),
                         prog_P_vs_identity_maxerr=float(np.abs(B['prog']['P'] - idw).max())), edits={})
log(f'seed 3 (N={len(x)}): network acc {(cbn == y).mean()*100:.2f} %, program acc {(cbp == y).mean()*100:.2f} %, '
    f'calls differ {(cbn != cbp).mean()*100:.2f} %; max |P_prog − identity(window)| {np.abs(B["prog"]["P"] - idw).max():.2e}')


# ---------------------------------------------------------------- edits
def side(xp):
    """(J, 9) sign of the 9 angular statements on the physical particle (float)"""
    return (xp @ alpha[:, ANG] + beta[ANG]) > 0


def side_net(xj, slot):
    """(J, 9) side of the 9 angular statements at one slot, on the network's rounded input"""
    xn = jnp.asarray(net.inputs(xj, 64))
    xq = np.asarray(L['l1'].iq(xn, training=False), np.float64)[np.arange(len(xj)), slot]
    return (xq @ K1[:, ANG] + b1[ANG]) > 0


def flip_and_rot(r):
    rr = np.hypot(x[:, r, 1], x[:, r, 2])
    J = np.flatnonzero((nreal > r) & (rr > 0.01))
    xa = x[J].copy(); xa[:, r, 1:] *= -1
    # rotation control: crossing angles of the 9 cut lines on the circle of radius |r|
    xc = x[J].copy(); keep = np.zeros(len(J), bool); dth = np.zeros(len(J))
    for n, j in enumerate(J):
        de, dp = x[j, r, 1], x[j, r, 2]; rad = np.hypot(de, dp); th = np.arctan2(dp, de)
        zs = []
        for c in ANG:
            a_e, a_p = alpha[1, c], alpha[2, c]; be = beta[c] + alpha[0, c] * x[j, r, 0]
            A = np.hypot(a_e, a_p); ph = np.arctan2(a_p, a_e); cz = -be / (rad * A)
            if abs(cz) <= 1: zs += [ph + np.arccos(cz), ph - np.arccos(cz)]
        if not zs: continue
        zs = np.array(zs)
        dplus = np.min((zs - th) % (2 * np.pi)); dminus = np.min((th - zs) % (2 * np.pi)); W = dplus + dminus
        m = max(0.1 * W, 0.004 / rad); rot = max(dplus, dminus) - m
        if rot <= 0: continue
        th2 = th + rot if dplus >= dminus else th - rot
        xc[n, r, 1], xc[n, r, 2] = rad * np.cos(th2), rad * np.sin(th2); keep[n] = True; dth[n] = rot
    s0 = side(x[J, r]); sf = side(xa[:, r]); sc = side(xc[:, r])
    sn0 = side_net(x[J], r); snf = side_net(xa, r); snc = side_net(xc, r)
    info_f = dict(n_side_changes_float=float((s0 != sf).sum(1).mean()), frac_all9=float(((s0 != sf).sum(1) == 9).mean()),
                  n_side_changes_net=float((sn0 != snf).sum(1).mean()), mean_r=float(rr[J].mean()),
                  displacement_mean=float((2 * rr[J]).mean()))
    Jc = J[keep]; xck = xc[keep]
    info_c = dict(n_side_changes_float=float((s0[keep] != sc[keep]).sum(1).mean()),
                  n_side_changes_net=float((sn0[keep] != snc[keep]).sum(1).mean()),
                  frac_any_net_side_change=float(((sn0[keep] != snc[keep]).sum(1) > 0).mean()),
                  rotation_deg_mean=float(np.degrees(dth[keep]).mean()), mean_r=float(rr[Jc].mean()),
                  displacement_mean=float((2 * rr[Jc] * np.sin(dth[keep] / 2)).mean()), skipped=int((~keep).sum()))
    clean = (sn0[keep] == snc[keep]).all(1)          # control jets with no side change on the network's rounded input
    info_cc = dict(info_c, n_side_changes_net=0.0, frac_any_net_side_change=0.0, subset='rotation jets with no side change '
                   'of the 9 statements on the rounded input', mean_r=float(rr[Jc[clean]].mean()))
    return (J, x[J], xa, info_f), (Jc, x[Jc], xck, info_c), (Jc[clean], x[Jc[clean]], xck[clean], info_cc)


def ramp(lo, hi, win):
    pt = x[..., 0]; J, slots = [], []
    nwin = 0
    for j in range(len(x)):
        n = nreal[j]; cand = np.flatnonzero((pt[j, :n] >= win[0]) & (pt[j, :n] <= win[1]))
        if len(cand): nwin += 1
        ok = [i for i in cand if (i == 0 or pt[j, i - 1] >= hi) and (pt[j, i + 1] <= lo if i + 1 < 64 else True)]
        if ok:
            i = min(ok, key=lambda i: abs(pt[j, i] - (lo + hi) / 2)); J.append(j); slots.append(i)
    J, slots = np.array(J), np.array(slots)
    xb = x[J].copy(); xa = x[J].copy()
    xb[np.arange(len(J)), slots, 0] = lo; xa[np.arange(len(J)), slots, 0] = hi
    seen = lambda xx: np.asarray(L['l1'].iq(jnp.asarray(net.inputs(xx, 64)), training=False), np.float64)[
        np.arange(len(J)), slots, 0] * net.scale[0] + net.shift[0]                 # pT as the rounded input encodes it
    step = seen(xa) - seen(xb)
    return J, xb, xa, dict(n_jets_with_particle_in_window=nwin, skipped_rank_change=nwin - len(J),
                           pt_step_seen_by_network_mean=float(step.mean()), frac_step_rounded_to_0=float((np.abs(step) < 1e-6).mean()),
                           pt_lsb_at_slot_mean=float((net.scale[0] * 2.0 ** -fbits[slots, 0]).mean()),
                           slot_mean=float(slots.mean()), slot_hist={int(s): int((slots == s).sum()) for s in np.unique(slots)})


def insert(xj, n, part):
    """insert particle part=(pT, Δη, Δφ) into the n real particles of xj at its pT rank (after equal pT)"""
    pos = int((xj[:n, 0] >= part[0]).sum())
    out = np.zeros_like(xj); out[:pos] = xj[:pos]; out[pos] = part; out[pos + 1:n + 1] = xj[pos:n]
    return out


def split_edit():
    J = np.flatnonzero(nreal < 64); xa = x[J].copy()
    for k, j in enumerate(J):
        n = nreal[j]; p = x[j, 0].copy(); h = np.array([p[0] / 2, p[1], p[2]], np.float32)
        rest = np.zeros_like(x[j]); rest[:n - 1] = x[j, 1:n]
        t = insert(rest, n - 1, h); xa[k] = insert(t, n, h)
    return J, x[J], xa, dict(skipped_64=int((nreal == 64).sum()))


def add_edit():
    J = np.flatnonzero(nreal < 64); xa = x[J].copy(); rng = np.random.default_rng(3)
    ph = rng.uniform(0, 2 * np.pi, len(J))
    for k, j in enumerate(J):
        xa[k] = insert(x[j], nreal[j], np.array([1.0, 0.1 * np.cos(ph[k]), 0.1 * np.sin(ph[k])], np.float32))
    return J, x[J], xa, dict(skipped_64=int((nreal == 64).sum()))


# ---------------------------------------------------------------- metrics
def r2_11(a, b):
    """1:1 R² of a (network) predicted by b (program / identity), no fit"""
    ss = ((a - a.mean()) ** 2).sum()
    return None if ss < 1e-12 else float(1 - ((a - b) ** 2).sum() / ss)


def corr(a, b):
    return None if a.std() < 1e-9 or b.std() < 1e-9 else float(np.corrcoef(a, b)[0, 1])


def boot_ratio(fa, fb, rng, n=1000):
    ra = [fa[rng.integers(0, len(fa), len(fa))].mean() / max(fb[rng.integers(0, len(fb), len(fb))].mean(), 1e-9) for _ in range(n)]
    return [float(np.percentile(ra, 2.5)), float(np.percentile(ra, 97.5))]


FLIPS = {}
def analyse(name, J, xb, xa, info):
    t = time.time()
    M = ['net', 'prog'] + HYB
    Rb = {m: run(xb, m) for m in M}; Ra = {m: run(xa, m) for m in M}
    d = {m: {k: Ra[m][k] - Rb[m][k] for k in ('P', 'S', 'logits')} for m in M}
    dPid = ident(xa, False) - ident(xb, False)                      # identity without the rank window
    out = dict(N=len(J), info=info, P={}, S={}, dlogit={})
    for c in LIVE_P:
        n_, p_, i_ = d['net']['P'][:, c], d['prog']['P'][:, c], dPid[:, c]
        out['P'][c] = dict(r2_net_prog=r2_11(n_, p_), corr=corr(n_, p_), r2_net_identity_nowindow=r2_11(n_, i_),
                           mean_abs_net=float(np.abs(n_).mean()), mean_abs_prog=float(np.abs(p_).mean()),
                           mean_abs_identity_nowindow=float(np.abs(i_).mean()), frac_changed_net=float((np.abs(n_) > 1e-9).mean()),
                           mean_net=float(n_.mean()), mean_prog=float(p_.mean()),
                           rel_rms_err=None if np.abs(p_).max() < 1e-9 else float(np.sqrt(((n_ - p_) ** 2).mean() / (p_ ** 2).mean())),
                           frac_changed_prog=float((np.abs(p_) > 1e-9).mean()))
    for c in LIVE_S:
        n_, p_ = d['net']['S'][:, c], d['prog']['S'][:, c]
        out['S'][c] = dict(r2_net_prog=r2_11(n_, p_), corr=corr(n_, p_), mean_abs_net=float(np.abs(n_).mean()),
                           mean_abs_prog=float(np.abs(p_).mean()))
    for k in range(5):
        n_, p_ = d['net']['logits'][:, k], d['prog']['logits'][:, k]
        out['dlogit'][k] = dict(corr=corr(n_, p_), r2_net_prog=r2_11(n_, p_), mean_abs_net=float(np.abs(n_).mean()),
                                mean_abs_prog=float(np.abs(p_).mean()))
    cnb, cna = Rb['net']['logits'].argmax(1), Ra['net']['logits'].argmax(1)
    cpb, cpa = Rb['prog']['logits'].argmax(1), Ra['prog']['logits'].argmax(1)
    fn, fp = cna != cnb, cpa != cpb; FLIPS[name] = (fn, fp)
    either = fn | fp
    pn, pp = fn.mean(), fp.mean()
    out['flip'] = dict(rate_net=float(pn), rate_prog=float(pp), se_net=float(np.sqrt(pn * (1 - pn) / len(J))),
                       agree_whether=float((fn == fp).mean()),
                       agree_whether_and_new_call=float(((fn == fp) & (~fn | (cna == cpa))).mean()),
                       agree_chance=float(pn * pp + (1 - pn) * (1 - pp)),
                       n_either=int(either.sum()),
                       both_same_new_call_given_either=float((fn & fp & (cna == cpa))[either].mean()) if either.any() else None,
                       base_calls_differ=float((cnb != cpb).mean()), edited_calls_differ=float((cna != cpa).mean()),
                       acc_net_before=float((cnb == y[J]).mean()), acc_net_after=float((cna == y[J]).mean()),
                       acc_prog_before=float((cpb == y[J]).mean()), acc_prog_after=float((cpa == y[J]).mean()))
    out['hybrid'] = {}
    for h in HYB:      # program + one group's fixed-point quantizers, compared with the network
        chb, cha = Rb[h]['logits'].argmax(1), Ra[h]['logits'].argmax(1); fh = cha != chb; eh = fn | fh
        out['hybrid'][h] = dict(dlogit_corr_vs_net=[corr(d['net']['logits'][:, k], d[h]['logits'][:, k]) for k in range(5)],
                                dS_corr_vs_net_median=float(np.median([v for v in (corr(d['net']['S'][:, c], d[h]['S'][:, c])
                                                                                    for c in LIVE_S) if v is not None] or [np.nan])),
                                flip_rate=float(fh.mean()), agree_whether_vs_net=float((fn == fh).mean()),
                                both_same_new_call_given_either=float((fn & fh & (cna == cha))[eh].mean()) if eh.any() else None)
    res['edits'][name] = out
    ang = [out['P'][c]['r2_net_prog'] for c in ANG if c in out['P'] and out['P'][c]['r2_net_prog'] is not None]
    dl = [out['dlogit'][k]['corr'] for k in range(5)]
    f = out['flip']
    log(f'\n[{name}] N={len(J)} {json.dumps(info)}  ({time.time()-t:.0f}s)')
    log(f'  flips: network {f["rate_net"]*100:.2f} %, program {f["rate_prog"]*100:.2f} %; agree on whether {f["agree_whether"]*100:.2f} % '
        f'(chance {f["agree_chance"]*100:.2f} %), whether+new call {f["agree_whether_and_new_call"]*100:.2f} %, '
        f'both flip to same call | either flips {f["both_same_new_call_given_either"] if f["both_same_new_call_given_either"] is None else round(f["both_same_new_call_given_either"]*100, 1)} % (n_either={f["n_either"]})')
    log(f'  dlogit corr net vs prog per class (g q W Z t): {[None if v is None else round(v, 3) for v in dl]}')
    log('  dP per live P unit: R²(net, prog) | R²(net, identity no window) | mean|dP| net, prog, no-window | frac changed net, prog')
    for c in LIVE_P:
        e = out['P'][c]
        if e['mean_abs_net'] < 1e-9 and e['mean_abs_prog'] < 1e-9 and e['mean_abs_identity_nowindow'] < 1e-9: continue
        fmt = lambda v: '  n/a' if v is None else f'{v:+.3f}'
        log(f'    P{c:<2d} {fmt(e["r2_net_prog"])} | {fmt(e["r2_net_identity_nowindow"])} | {e["mean_abs_net"]:.4f} {e["mean_abs_prog"]:.4f} '
            f'{e["mean_abs_identity_nowindow"]:.4f} | {e["frac_changed_net"]:.2f} {e["frac_changed_prog"]:.2f}')
    sr = [(c, out['S'][c]['r2_net_prog']) for c in LIVE_S]
    log('  dS R²(net, prog) per live S unit:', ', '.join(f'S{c}:{"n/a" if v is None else round(v, 3)}' for c, v in sr))
    if ang: log(f'  angular P units R² (net, prog): min {min(ang):.3f}, median {np.median(ang):.3f} over {len(ang)} units')
    for c in (15, 32, 56):
        e = out['P'][c]
        if e['mean_abs_prog'] > 1e-9 or e['mean_abs_net'] > 1e-9:
            log(f'  P{c}: mean dP net {e["mean_net"]:+.5f}, prog {e["mean_prog"]:+.5f}, rms(net−prog)/rms(prog) '
                f'{"n/a" if e["rel_rms_err"] is None else round(e["rel_rms_err"], 3)}')
    for h, e in out['hybrid'].items():
        log(f'  {h:7s} vs network: dlogit corr {[None if v is None else round(v, 3) for v in e["dlogit_corr_vs_net"]]}, '
            f'median dS corr {e["dS_corr_vs_net_median"]:.3f}, flips {e["flip_rate"]*100:.2f} %, same flip | either '
            f'{"n/a" if e["both_same_new_call_given_either"] is None else round(e["both_same_new_call_given_either"]*100, 1)} %')


for r in (0, 2, 13, 14, 30):
    (J, xb, xa, inf), (Jc, xbc, xac, infc), (Jk, xbk, xak, infk) = flip_and_rot(r)
    analyse(f'flip_r{r}', J, xb, xa, inf); analyse(f'rot_r{r}', Jc, xbc, xac, infc)
    analyse(f'rot_r{r}_netsame', Jk, xbk, xak, infk)
for name, lo, hi, win in [('a32_x', 10., 12., (9, 13)), ('a32_c_lo', 5., 7., (4, 8)), ('a32_c_hi', 13., 15., (12, 16)),
                          ('a56_x', 30., 32., (29, 33)), ('a56_c_lo', 25., 27., (24, 28)), ('a56_c_hi', 33., 35., (32, 36))]:
    J, xb, xa, inf = ramp(lo, hi, win); analyse(name, J, xb, xa, inf)
analyse('split', *split_edit()); analyse('add1', *add_edit())

# ---------------------------------------------------------------- summary: crossing / control flip-rate ratios, pooled agreement
rng = np.random.default_rng(3); res['ratios'] = {}
pairs = [(f'flip_r{r}', f'rot_r{r}') for r in (0, 2, 13, 14, 30)] + [(f'flip_r{r}', f'rot_r{r}_netsame') for r in (0, 2, 13, 14, 30)] + [('a32_x', 'a32_c_lo'), ('a32_x', 'a32_c_hi'),
                                                                      ('a56_x', 'a56_c_lo'), ('a56_x', 'a56_c_hi')]
log('\nflip-rate ratio crossing / control (network; program), 95 % bootstrap CI (1000 resamples of jets)')
for a, b in pairs:
    e = {}
    for m, k in (('net', 0), ('prog', 1)):
        fa, fb = FLIPS[a][k].astype(float), FLIPS[b][k].astype(float)
        e[m] = dict(ratio=float(fa.mean() / max(fb.mean(), 1e-9)), ci=boot_ratio(fa, fb, rng), cross=float(fa.mean()), ctrl=float(fb.mean()))
    res['ratios'][f'{a}/{b}'] = e
    log(f'  {a:9s}/{b:9s} net {e["net"]["cross"]*100:5.2f}/{e["net"]["ctrl"]*100:5.2f} % = {e["net"]["ratio"]:.2f} '
        f'[{e["net"]["ci"][0]:.2f}, {e["net"]["ci"][1]:.2f}];  prog {e["prog"]["cross"]*100:5.2f}/{e["prog"]["ctrl"]*100:5.2f} % = '
        f'{e["prog"]["ratio"]:.2f} [{e["prog"]["ci"][0]:.2f}, {e["prog"]["ci"][1]:.2f}]')
PRIMARY = [k for k in FLIPS if not k.endswith('_netsame')]
fn = np.concatenate([FLIPS[k][0] for k in PRIMARY]); fp = np.concatenate([FLIPS[k][1] for k in PRIMARY])
res['pooled'] = dict(N=int(len(fn)), agree_whether=float((fn == fp).mean()), rate_net=float(fn.mean()), rate_prog=float(fp.mean()),
                     chance=float(fn.mean() * fp.mean() + (1 - fn.mean()) * (1 - fp.mean())),
                     both_flip_given_either=float((fn & fp)[fn | fp].mean()))
log(f'\npooled over the {len(PRIMARY)} primary edits: N={len(fn)} edited jets, flips net {fn.mean()*100:.2f} %, prog {fp.mean()*100:.2f} %, '
    f'agree on whether {(fn == fp).mean()*100:.2f} % (chance {res["pooled"]["chance"]*100:.2f} %), both flip | either flips '
    f'{res["pooled"]["both_flip_given_either"]*100:.1f} %')

# ---------------------------------------------------------------- stage-by-stage agreement (figure values; plotted by e13_figure.py)
GRP = {'reflection through the axis (5 ranks)': [f'flip_r{r}' for r in (0, 2, 13, 14, 30)],
       'rotation control, same sides (5 ranks)': [f'rot_r{r}' for r in (0, 2, 13, 14, 30)],
       '2 GeV pT steps (a32, a56; 6 edits)': ['a32_x', 'a32_c_lo', 'a32_c_hi', 'a56_x', 'a56_c_lo', 'a56_c_hi'],
       'count +1: split, add 1 GeV (2 edits)': ['split', 'add1']}
def stage_vals(e):
    pv = [v['corr'] for v in e['P'].values() if v['corr'] is not None]
    sv = [v['corr'] for v in e['S'].values() if v['corr'] is not None]
    lv = [v['corr'] for v in e['dlogit'].values() if v['corr'] is not None]
    return [np.median(pv) if pv else np.nan, np.median(sv) if sv else np.nan, np.mean(lv)]
res['figure_values'] = {}
log('\nstage-by-stage agreement of the edit response (Pearson r, network vs program): median live P | median live S | mean logits')
for g, ks in GRP.items():
    per = np.array([stage_vals(res['edits'][k]) for k in ks]); v = np.nanmedian(per, 0)
    res['figure_values'][g] = dict(median=v.tolist(), per_edit={k: per[i].tolist() for i, k in enumerate(ks)})
    log(f'  {g:42s} {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f}')
(OUT / 'e13_edit_fidelity.json').write_text(json.dumps(res, indent=1))
log(f'done {time.time()-t0:.0f}s')
