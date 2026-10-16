"""E11: are the HGQ zero-bit slot masks (pT-rank windows) functional parts of the second-round objects?

Programs (same arithmetic as interp/e3m_masks.py):
  fixed  the fixed-point network Jedi(64) (jedi_core forward)
  mask   float arithmetic, inputs of l1, qs, l2, dd, l3, qs1, d4, d5, d6, d7 multiplied by (iq.quantizer.b > 0)
         (E3m mk('mask'); slot x channel masks on qs, l2, l3, qs1; per-channel masks on dd, d4, d5)
  float  float arithmetic, no masks
Parts (sample: sample(per_class=2000, seed) for seed in SEEDS):
  (a) per layer (qs, l2, l3, qs1) and channel: share of the channel's input activation mass sum_{masked slots}|t| /
      sum_all|t| (t = layer input in the mask program; also with upstream masks off, and real-particle slots only)
  (b) mask program with one layer's slot mask set to 1 (4 runs), one channel's slot mask set to 1 (per layer, every
      channel with nonzero input mass), all four slot masks set to 1, dd/d4/d5 masks set to 1
  (c) qs1 contiguous windows: one edge moved by d in {-4,-2,-1,+1,+2,+4} slots, one channel at a time; matched control:
      the same number of mask entries flipped at random slots of that channel (5 draws)
  (d) slot masks of qs, l2, l3, qs1 permuted along the slot axis per channel (rng 0..9; jointly and one layer at a time);
      reversed in slot order; cyclic roll by r slots; scale-only (each channel's mask replaced by kept/64 on every slot)
  (e) slots k, k+1 swapped in jets where both are real particles (k = 0..62): % calls changed and mean |dS_c|/sd(S_c)
      per live S channel in float, mask and fixed programs
"""
import sys, json, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '.')
from interp.jedi_core import Jedi, sample

T0 = time.time()
LOG = open('logs/e11.log', 'w')


def log(*a):
    s = ' '.join(str(v) for v in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


SEEDS = [int(v) for v in sys.argv[1:]] or [0, 1]
net = Jedi(64); L = net.L
KB = {k: (jnp.asarray(v[0], jnp.float32), jnp.asarray(v[1], jnp.float32)) for k, v in net.KB.items()}
NAMES = ('l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7')
SLOT = ('qs', 'l2', 'l3', 'qs1')                     # slot x channel masks, shape (1, 64, 64)
MASK0 = {k: (np.asarray(L[k].iq.quantizer.b) > 0).astype(np.float32) for k in NAMES}
ONES = {k: np.ones_like(v) for k, v in MASK0.items()}
for k in NAMES:
    log(f'mask {k:4s} shape {MASK0[k].shape} zero-bit {100 * (1 - MASK0[k].mean()):.1f}%')

E1 = json.load(open('results/jedi/e1_e3.json'))['E1']
live = {st: [c for c, v in enumerate(E1[st]['changed']) if v >= 0.01] for st in ('a', 's', 'g', 'u', 'S')}
LIVE = {'qs': live['a'], 'l2': live['a'], 'l3': sorted(set(live['s']) | set(live['g'])), 'qs1': live['u']}
LIVE_S = live['S']
log('live channels (E1, >=1 % calls):', {k: v for k, v in LIVE.items()}, 'live S', LIVE_S)


def fwd(xn, M):
    def q(k, t): return t * M[k]

    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z

    def pool(k, a): return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
    u = dense('l3', s + g); S = pool('qs1', u)
    lo = dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))
    J = xn.shape[0]
    return lo.reshape(J, -1), S.reshape(J, -1)


def mass(xn, M, real):
    """per (slot, channel) sums of |layer input| over jets, all slots and real-particle slots"""
    def q(k, t): return t * M[k]

    def dense(k, t):
        return jax.nn.relu(jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1])
    a = dense('l1', xn); P = jnp.sum(q('qs', a), axis=L['qs'].axes, keepdims=L['qs'].keepdims) * L['qs'].scale
    g = dense('dd', P); s = dense('l2', a); t = s + g; u = dense('l3', t)
    out = {}
    for k, T in (('qs', a), ('l2', a), ('l3', t), ('qs1', u)):
        A = jnp.abs(T)
        out[k] = (A.sum(0), (A * real[:, :, None]).sum(0), (T.max(1) - T.min(1)).max(0))
    return out


F, FM = jax.jit(fwd), jax.jit(mass)
B = 5000


def run(xn, M):
    Mj = {k: jnp.asarray(v) for k, v in M.items()}
    lo, S = [], []
    for i in range(0, len(xn), B):
        l_, s_ = F(xn[i:i + B], Mj); lo.append(np.asarray(l_, np.float64)); S.append(np.asarray(s_, np.float64))
    return np.concatenate(lo), np.concatenate(S)


def run_fixed(x):
    r = net.run(x, keep=('logits', 'S'), batch=B)
    return r['logits'], r['S']


def masses(xn, real, M):
    Mj = {k: jnp.asarray(v) for k, v in M.items()}
    acc = {k: [np.zeros((64, 64)), np.zeros((64, 64)), np.zeros(64)] for k in SLOT}
    for i in range(0, len(xn), B):
        r = FM(xn[i:i + B], Mj, jnp.asarray(real[i:i + B], jnp.float32))
        for k in SLOT:
            acc[k][0] += np.asarray(r[k][0], np.float64); acc[k][1] += np.asarray(r[k][1], np.float64)
            acc[k][2] = np.maximum(acc[k][2], np.asarray(r[k][2], np.float64))
    return acc


def windows():
    """contiguous qs1 windows of the live u channels"""
    W = {}
    for c in LIVE['qs1']:
        kept = np.flatnonzero(MASK0['qs1'][0, :, c] > 0)
        if len(kept) == 64 or len(kept) == 0: continue
        lo, hi = int(kept.min()), int(kept.max())
        if len(kept) == hi - lo + 1: W[c] = (lo, hi)
    return W


def describe(col):
    kept = np.flatnonzero(col > 0)
    if len(kept) == 0: return 'none'
    runs, st = [], kept[0]
    for i in range(1, len(kept) + 1):
        if i == len(kept) or kept[i] != kept[i - 1] + 1:
            runs.append(f'{st}-{kept[i - 1]}' if kept[i - 1] > st else f'{st}'); st = kept[i] if i < len(kept) else None
    return ','.join(runs)


def with_(base, k, arr):
    M = dict(base); M[k] = arr.astype(np.float32); return M


def experiment(seed, full):
    t0 = time.time()
    x, y, _ = sample(per_class=2000, seed=seed)
    xn = net.inputs(x, 64); nreal = (x[:, :, 0] > 0).sum(1); real = (np.arange(64)[None] < nreal[:, None]).astype(np.float32)
    R = {'seed': seed, 'N': int(len(y)), 'per_class': 2000}
    lo_f, S_f = run_fixed(x); cf = lo_f.argmax(1)
    lo_m, S_m = run(xn, MASK0); cm = lo_m.argmax(1)
    lo_0, S_0 = run(xn, ONES); c0 = lo_0.argmax(1)

    def st(lo):
        c = lo.argmax(1)
        return dict(acc=round(float((c == y).mean()), 5), chg_vs_mask=round(100 * float((c != cm).mean()), 3),
                    chg_vs_fixed=round(100 * float((c != cf).mean()), 3))
    R['base'] = {'fixed': st(lo_f), 'mask': st(lo_m), 'float': st(lo_0)}
    log(f'\n=== seed {seed}: N={len(y)} ===  fixed acc {R["base"]["fixed"]["acc"]:.4f} | mask acc {R["base"]["mask"]["acc"]:.4f} '
        f'({R["base"]["mask"]["chg_vs_fixed"]:.2f}% calls vs fixed) | float acc {R["base"]["float"]["acc"]:.4f} '
        f'({R["base"]["float"]["chg_vs_fixed"]:.2f}%)')

    # (a) activation mass removed by each mask
    A = {}
    for up, M in (('mask_upstream', MASK0), ('no_mask_upstream', ONES)):
        ms = masses(xn, real, M); A[up] = {}
        for k in SLOT:
            m0 = MASK0[k][0]; tot, totr, rg = ms[k]
            ch = {}
            for c in range(64):
                T, Tr = tot[:, c].sum(), totr[:, c].sum()
                if T <= 0: continue
                ch[c] = dict(removed=round(float(tot[m0[:, c] == 0, c].sum() / T), 5),
                             removed_real=round(float(totr[m0[:, c] == 0, c].sum() / Tr), 5) if Tr > 0 else None,
                             slot_frac_masked=round(float(1 - m0[:, c].mean()), 4), live=c in LIVE[k],
                             jet_level=bool(rg[c] < 1e-6))
                if up == 'mask_upstream' and k == 'qs1' and c in LIVE[k]:
                    ch[c]['slot_profile'] = [round(float(v / T), 5) for v in tot[:, c]]
            lv = LIVE[k]
            agg = float(sum((tot[m0[:, c] == 0, c]).sum() for c in lv) / max(sum(tot[:, c].sum() for c in lv), 1e-30))
            agg_all = float((tot * (m0 == 0)).sum() / tot.sum())
            A[up][k] = dict(channels=ch, live_mass_weighted=round(agg, 5), all_mass_weighted=round(agg_all, 5))
    # qs1 mask's removed share when one upstream slot-mask layer is set to 1 (which upstream mask zeroes u off-window)
    A['qs1_removed_upstream_unmasked'] = {}
    for up in ('qs', 'l2', 'l3', 'qs_l2_l3'):
        M = dict(MASK0)
        for k in (('qs', 'l2', 'l3') if up == 'qs_l2_l3' else (up,)): M[k] = ONES[k]
        tot = masses(xn, real, M)['qs1'][0]; m0 = MASK0['qs1'][0]
        A['qs1_removed_upstream_unmasked'][up] = {c: round(float(tot[m0[:, c] == 0, c].sum() / tot[:, c].sum()), 5)
                                                  for c in LIVE['qs1'] if tot[:, c].sum() > 0}
    R['a'] = A
    log('(a) share of input activation mass on masked slots (mask program upstream; all slots | real slots | masked slot fraction)')
    for k in SLOT:
        a_ = A['mask_upstream'][k]
        log(f'  {k:4s} live channels mass-weighted {100 * a_["live_mass_weighted"]:.2f}%, all channels {100 * a_["all_mass_weighted"]:.2f}%'
            f' | no-mask upstream: live {100 * A["no_mask_upstream"][k]["live_mass_weighted"]:.2f}%')
        for c in LIVE[k]:
            v = a_['channels'].get(c)
            if v is None: log(f'    ch{c:2d}: zero input mass'); continue
            vn = A['no_mask_upstream'][k]['channels'].get(c, {}).get('removed', 0)
            log(f'    ch{c:2d}: {100 * v["removed"]:6.2f}% | {100 * (v["removed_real"] or 0):6.2f}% | {100 * v["slot_frac_masked"]:5.1f}%'
                f' | no-mask upstream {100 * vn:6.2f}%{"  jet-level" if v["jet_level"] else ""}  kept {describe(MASK0[k][0, :, c])}')
        ex = [c for c, v in a_['channels'].items() if c not in LIVE[k] and v['removed'] > 0]
        if ex: log(f'    non-live channels with masked mass: ' + ', '.join(
            f'ch{c} {100 * a_["channels"][c]["removed"]:.1f}%{" jet-level" if a_["channels"][c]["jet_level"] else ""} kept {describe(MASK0[k][0, :, c])}' for c in ex))
    for up, v in A['qs1_removed_upstream_unmasked'].items():
        log(f'  qs1 removed share with {up} slot masks set to 1: ' + ', '.join(f'u{c} {100 * r:.1f}%' for c, r in v.items()))

    # (b) unmasking
    Bres = {'layer': {}, 'channel': {k: {} for k in SLOT}}
    for k in SLOT:
        Bres['layer'][k] = st(run(xn, with_(MASK0, k, ONES[k]))[0])
    M = dict(MASK0); [M.__setitem__(k, ONES[k]) for k in SLOT]; Bres['layer']['all_slot_masks'] = st(run(xn, M)[0])
    M = dict(MASK0); [M.__setitem__(k, ONES[k]) for k in ('dd', 'd4', 'd5')]; Bres['layer']['dd_d4_d5'] = st(run(xn, M)[0])
    log('(b) unmask: acc | % calls changed vs mask | vs fixed')
    import itertools
    Bres['subsets_kept'] = {}
    for r_ in range(5):
        for keep in itertools.combinations(SLOT, r_):
            M = dict(MASK0)
            for k in SLOT:
                if k not in keep: M[k] = ONES[k]
            Bres['subsets_kept']['+'.join(keep) or 'none'] = st(run(xn, M)[0])
    for k, v in Bres['layer'].items():
        log(f'  layer {k:14s} acc {v["acc"]:.4f} | {v["chg_vs_mask"]:6.2f}% | {v["chg_vs_fixed"]:6.2f}%')
    log('  only these layers keep their slot masks (dd/d4/d5 masks kept): acc | % calls changed vs mask')
    for k, v in Bres['subsets_kept'].items():
        log(f'    {k:16s} acc {v["acc"]:.4f} | {v["chg_vs_mask"]:6.2f}%')
    for k in SLOT:
        chans = [c for c in range(64) if c in A['mask_upstream'][k]['channels'] and MASK0[k][0, :, c].min() == 0]
        if not full: chans = [c for c in chans if c in LIVE[k]]
        for c in chans:
            arr = MASK0[k].copy(); arr[0, :, c] = 1
            Bres['channel'][k][c] = dict(st(run(xn, with_(MASK0, k, arr))[0]), live=c in LIVE[k])
        top = sorted(Bres['channel'][k].items(), key=lambda kv: -kv[1]['chg_vs_mask'])
        log(f'  per channel {k}: {len(chans)} channels run; top by calls changed: ' +
            ', '.join(f'ch{c} {v["chg_vs_mask"]:.2f}% (acc {v["acc"]:.4f})' for c, v in top[:8]))
        if k == 'qs1':
            log('  qs1 live channels: ' + ', '.join(f'u{c} {Bres["channel"][k][c]["chg_vs_mask"]:.2f}%' for c in LIVE[k]
                                                   if c in Bres['channel'][k]))
    R['b'] = Bres

    # (c) edge shifts of contiguous qs1 windows + matched random flips
    W = windows(); C = {}
    rng = np.random.default_rng(1000 + seed)
    for c, (lo, hi) in W.items():
        edge = 'hi' if lo == 0 else 'lo'
        C[c] = {'window': [lo, hi], 'edge': edge, 'shifts': {}}
        col0 = MASK0['qs1'][0, :, c]
        for d in (-4, -2, -1, 1, 2, 4):
            if edge == 'hi': nlo, nhi = 0, int(np.clip(hi + d, -1, 63))
            else: nlo, nhi = int(np.clip(lo + d, 0, 64)), 63
            col = np.zeros(64, np.float32); col[nlo:nhi + 1] = 1
            nflip = int((col != col0).sum())
            arr = MASK0['qs1'].copy(); arr[0, :, c] = col
            r = dict(st(run(xn, with_(MASK0, 'qs1', arr))[0]), new_window=[nlo, nhi], n_flipped=nflip)
            ctrl = []
            if nflip > 0:
                grow = col.sum() > col0.sum()
                pool_ = np.flatnonzero(col0 == 0) if grow else np.flatnonzero(col0 > 0)
                for _ in range(5):
                    pick = rng.choice(pool_, nflip, replace=False)
                    cc = col0.copy(); cc[pick] = 1.0 if grow else 0.0
                    arr = MASK0['qs1'].copy(); arr[0, :, c] = cc
                    ctrl.append(st(run(xn, with_(MASK0, 'qs1', arr))[0])['chg_vs_mask'])
            r['random_flip_chg_vs_mask'] = [round(v, 3) for v in ctrl]
            C[c]['shifts'][d] = r
    R['c'] = C
    log('(c) qs1 window edge shift: % calls changed vs mask program [matched random flips of the same count, mean of 5]')
    for c, v in C.items():
        log(f'  u{c:2d} window {v["window"][0]}-{v["window"][1]} ({v["edge"]} edge): ' + '  '.join(
            f'{d:+d}:{r["chg_vs_mask"]:.2f}[{np.mean(r["random_flip_chg_vs_mask"]) if r["random_flip_chg_vs_mask"] else 0:.2f}]'
            for d, r in v['shifts'].items()))

    # (d) random / reversed / rolled / scale-only slot masks
    D = {}

    def perm_mask(k, r_):
        arr = MASK0[k].copy()
        for c in range(64): arr[0, :, c] = arr[0, r_.permutation(64), c]
        return arr
    for scope in (('joint',) + SLOT):
        layers = SLOT if scope == 'joint' else (scope,)
        accs, chg = [], []
        for draw in range(10):
            r_ = np.random.default_rng(draw); M = dict(MASK0)
            for k in layers: M[k] = perm_mask(k, r_)
            s_ = st(run(xn, M)[0]); accs.append(s_['acc']); chg.append(s_['chg_vs_mask'])
        M = dict(MASK0)
        for k in layers: M[k] = MASK0[k][:, ::-1, :].copy()
        rev = st(run(xn, M)[0])
        M = dict(MASK0)
        for k in layers: M[k] = np.broadcast_to(MASK0[k].mean(1, keepdims=True), MASK0[k].shape).astype(np.float32)
        scl = st(run(xn, M)[0])
        D[scope] = dict(perm_acc=accs, perm_acc_mean=round(float(np.mean(accs)), 5), perm_acc_sd=round(float(np.std(accs, ddof=1)), 5),
                        perm_chg_vs_mask_mean=round(float(np.mean(chg)), 3), reversed=rev, scale_only=scl)
        if scope == 'joint':
            D['roll'] = {}
            for rr in (-8, -4, -2, -1, 1, 2, 4, 8):
                M = dict(MASK0)
                for k in layers: M[k] = np.roll(MASK0[k], rr, axis=1)
                D['roll'][rr] = st(run(xn, M)[0])
    R['d'] = D
    log('(d) replaced slot masks: acc (calls changed vs mask) | trained mask acc '
        f'{R["base"]["mask"]["acc"]:.4f}, no masks {R["base"]["float"]["acc"]:.4f}')
    for scope in (('joint',) + SLOT):
        v = D[scope]
        log(f'  {scope:5s} permuted per channel (10 draws) {100 * v["perm_acc_mean"]:.2f} +- {100 * v["perm_acc_sd"]:.2f}% '
            f'({v["perm_chg_vs_mask_mean"]:.1f}%) | reversed {100 * v["reversed"]["acc"]:.2f}% ({v["reversed"]["chg_vs_mask"]:.1f}%)'
            f' | scale-only {100 * v["scale_only"]["acc"]:.2f}% ({v["scale_only"]["chg_vs_mask"]:.1f}%)')
    log('  joint cyclic roll: ' + '  '.join(f'{r:+d}:{100 * v["acc"]:.2f}%' for r, v in D['roll'].items()))

    # (e) adjacent-slot swaps
    sd = {'float': S_0.std(0), 'mask': S_m.std(0), 'fixed': S_f.std(0)}
    base = {'float': (c0, S_0), 'mask': (cm, S_m), 'fixed': (cf, S_f)}
    E = {'live_S': LIVE_S, 'k': {}}
    for k in range(63):
        el = nreal >= k + 2
        xs = x.copy(); xs[el, k], xs[el, k + 1] = x[el, k + 1], x[el, k]
        xsn = net.inputs(xs, 64)
        out = {'float': run(xsn, ONES), 'mask': run(xsn, MASK0), 'fixed': run_fixed(xs)}
        ek = {'N': int(el.sum())}
        for p, (lo_s, S_s) in out.items():
            cb, Sb = base[p]
            dS = np.abs(S_s[el] - Sb[el])
            ek[p] = dict(chg=round(100 * float((lo_s[el].argmax(1) != cb[el]).mean()), 3),
                         dS_over_sd={c: round(float(dS[:, c].mean() / sd[p][c]), 5) if sd[p][c] > 0 else None for c in LIVE_S},
                         max_abs_dS=float(dS.max()))
        E['k'][k] = ek
    # mask edges between slots k and k+1 (channels with nonzero input mass in the mask program), per layer
    E['edges'] = {}
    for k in range(63):
        ed = []
        for nm in SLOT:
            m = MASK0[nm][0] > 0
            for c in A['mask_upstream'][nm]['channels']:
                if m[k, c] != m[k + 1, c]: ed.append(f'{nm}:{c}{"*" if c in LIVE[nm] else ""}')
        E['edges'][k] = ed
    for tag, sel in (('any', lambda ed: len(ed) > 0), ('live', lambda ed: any(e_.endswith('*') for e_ in ed)),
                     ('contiguous_qs1_or_l3', lambda ed: any(e_.startswith(('qs1:', 'l3:')) and e_.endswith('*') for e_ in ed))):
        ks = [k for k in range(63) if sel(E['edges'][k])]; ko = [k for k in range(63) if k not in ks]
        E[f'edge_summary_{tag}'] = dict(
            n_edge_k=len(ks), n_other_k=len(ko),
            mask_chg_edge_mean=round(float(np.mean([E['k'][k]['mask']['chg'] for k in ks])), 4) if ks else None,
            mask_chg_other_mean=round(float(np.mean([E['k'][k]['mask']['chg'] for k in ko])), 4) if ko else None,
            mask_chg_other_max=round(float(np.max([E['k'][k]['mask']['chg'] for k in ko])), 4) if ko else None,
            fixed_chg_edge_mean=round(float(np.mean([E['k'][k]['fixed']['chg'] for k in ks])), 4) if ks else None,
            fixed_chg_other_mean=round(float(np.mean([E['k'][k]['fixed']['chg'] for k in ko])), 4) if ko else None)
    R['e'] = E
    log('(e) swap slots k,k+1 (both real): N | % calls changed float / mask / fixed | max|dS| float')
    for k, v in E['k'].items():
        log(f'  k={k:2d} N={v["N"]:5d} | {v["float"]["chg"]:5.2f} {v["mask"]["chg"]:6.2f} {v["fixed"]["chg"]:6.2f} | {v["float"]["max_abs_dS"]:.1e}')
    for k in range(63):
        if E['k'][k]['mask']['chg'] >= 0.05: log(f'  mask edges at k={k}/{k + 1} (chg {E["k"][k]["mask"]["chg"]:.2f}%): {" ".join(E["edges"][k])}')
    for tag in ('any', 'live', 'contiguous_qs1_or_l3'):
        log(f'  edge k ({tag} channel edge) vs other k: {E[f"edge_summary_{tag}"]}')
    log(f'seed {seed} done in {time.time() - t0:.0f}s')
    return R


OUT = {'windows_qs1': {c: describe(MASK0['qs1'][0, :, c]) for c in LIVE['qs1']},
       'windows_qs_live': {c: describe(MASK0['qs'][0, :, c]) for c in LIVE['qs']},
       'live': LIVE, 'live_S': LIVE_S}
log('qs1 kept slots per live u channel:', OUT['windows_qs1'])
for s in SEEDS:
    OUT[f'seed{s}'] = experiment(s, full=(s == SEEDS[0]))
    json.dump(OUT, open('results/jedi/e11_windows.json', 'w'), default=lambda o: o.item() if hasattr(o, 'item') else str(o))
log(f'done {time.time() - T0:.0f}s')
