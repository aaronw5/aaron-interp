"""E20v part 2: the 'add' layer's zero-bit masks (on s per slot x channel, 90.7 % zero; on g per channel, 82.8 % zero) are not in
the E20 / E3m float + masks program. Does adding them (i) bring the program closer to the fixed-point network (calls, per-unit dS
R^2, S30's angle response), (ii) act as a further window layer (does the W13 peak of S43/S52/S63 survive 'all four slot masks = 1'
when the add masks are kept)?
Sample: interp.jedi_core.sample(per_class=2000, seed=77). Writes $AI_RESULTS/jedi/e20_verify_add.json.
Run: python -m interp.e20_verify_add
"""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
from interp import e20_rank_windows_jedi as E
from interp.jedi_core import sample

SEED = 77; PER = 2000; T0 = time.time()
def log(*a): print(f'[a {time.time() - T0:7.1f}s]', *a, flush=True)
L = E.L; LIVE = E.LIVE_S; SLOT = E.SLOT; MASK0 = E.MASK0
qa = L['add'].iq.quantizers
BADD = [np.asarray((q.quantizer if hasattr(q, 'quantizer') else q).b) for q in qa]
MA_S = (BADD[0] > 0).astype(np.float32); MA_G = (BADD[1] > 0).astype(np.float32)
log('add s mask', MA_S.shape, 'zero', float((MA_S == 0).mean()), '| add g mask', MA_G.shape, 'zero', float((MA_G == 0).mean()))


def fwd(xn, M, KB, ms, mg):
    def q(k, t): return t * M[k]
    def dense(k, t, act=True):
        z = jnp.einsum(L[k].equation, q(k, t), KB[k][0]) + KB[k][1]
        return jax.nn.relu(z) if act and L[k].activation is not None else z
    def pool(k, a): return jnp.sum(q(k, a), axis=L[k].axes, keepdims=L[k].keepdims) * L[k].scale
    a = dense('l1', xn); g = dense('dd', pool('qs', a)); s = dense('l2', a)
    u = dense('l3', s * ms + g * mg); S = pool('qs1', u)
    lo = dense('d7', jax.nn.relu(dense('d6', dense('d5', dense('d4', S)), act=False)))
    J = xn.shape[0]
    return lo.reshape(J, -1), S.reshape(J, -1)


F = jax.jit(fwd)


def run(x, M, ms, mg):
    Mj = {k: jnp.asarray(v) for k, v in M.items()}; lo, S = [], []
    for i in range(0, len(x), 5000):
        l_, s_ = F(jnp.asarray(E.net.inputs(x[i:i + 5000], 64)), Mj, E.KBT, jnp.asarray(ms), jnp.asarray(mg))
        lo.append(np.asarray(l_, np.float64)); S.append(np.asarray(s_, np.float64))
    return np.concatenate(lo), np.concatenate(S)


def main():
    R = dict(seed=SEED, per_class=PER)
    R['add_s_kept_slots'] = {f's{c}': E.describe(MA_S[0][:, c]) for c in (43, 60, 53, 32, 52, 14)}
    R['add_g_kept'] = np.flatnonzero(MA_G.reshape(-1) > 0).tolist()
    log('add s kept slots:', R['add_s_kept_slots']); log('add g kept channels:', R['add_g_kept'])
    x, y, _ = sample(per_class=PER, seed=SEED); nreal = (x[..., 0] > 0).sum(1)
    Jc = np.flatnonzero(nreal > 17); xs = x[Jc]
    bn = E.net.run(x, keep=('logits', 'S')); cnet = bn['logits'].argmax(1)
    one_s = np.ones_like(MA_S); one_g = np.ones_like(MA_G)
    def ones(layers):
        M = dict(MASK0); M.update({k: np.ones_like(MASK0[k]) for k in layers}); return M
    PR = {'mask_noadd': (dict(MASK0), one_s, one_g), 'mask_add': (dict(MASK0), MA_S, MA_G),
          'ones_all4_keepadd': (ones(SLOT), MA_S, MA_G), 'ones_all4_add_s_ones': (ones(SLOT), one_s, MA_G),
          'ones_l3_qs1_keepadd': (ones(('l3', 'qs1')), MA_S, MA_G)}
    base = {}; R['programs'] = {}
    for p, (M, ms, mg) in PR.items():
        lo, S = run(x, M, ms, mg); base[p] = (lo, S)
        R['programs'][p] = dict(acc=float((lo.argmax(1) == y).mean()), calls_vs_net=float((lo.argmax(1) != cnet).mean()),
                                sd_S={c: float(S[:, c].std()) for c in LIVE},
                                r2_S_vs_net={c: E.r2(bn['S'][:, c], S[:, c]) for c in LIVE})
        log(f'{p}: acc {100 * R["programs"][p]["acc"]:.2f} % calls vs net {100 * R["programs"][p]["calls_vs_net"]:.2f} %')
    R['acc_net'] = float((cnet == y).mean())
    R['edits'] = {}
    for k in (0, 5, 12, 13, 14):
        J, xe, _ = E.make_edit(xs, 'swap', k)
        r = E.net.run(xe, keep=('logits', 'S')); dn = r['S'] - bn['S'][Jc][J]
        rec = dict(N=int(len(J)), net=dict(calls=float((r['logits'].argmax(1) != cnet[Jc][J]).mean()),
                                           mean_abs={c: float(np.abs(dn[:, c]).mean()) for c in LIVE}))
        for p, (M, ms, mg) in PR.items():
            lo, S = run(xe, M, ms, mg); d = S - base[p][1][Jc][J]
            rec[p] = dict(calls=float((lo.argmax(1) != base[p][0][Jc][J].argmax(1)).mean()),
                          mean_abs={c: float(np.abs(d[:, c]).mean()) for c in LIVE},
                          r2_vs_net={c: E.r2(dn[:, c], d[:, c]) for c in LIVE})
        R['edits'][f'W{k}'] = rec
        log(f'W{k}: ' + ' | '.join(f'S{c} net {rec["net"]["mean_abs"][c]:.4f} ' + ' '.join(
            f'{p[:14]} {rec[p]["mean_abs"][c]:.4f} (R2 {rec[p]["r2_vs_net"][c] if rec[p]["r2_vs_net"][c] is None else round(rec[p]["r2_vs_net"][c], 2)})'
            for p in PR) for c in (43, 52, 63, 30, 17, 1)))
    R['wall_s'] = time.time() - T0
    json.dump(R, open(E.OUT / 'e20_verify_add.json', 'w'), indent=1, default=float); log('wrote', E.OUT / 'e20_verify_add.json')


if __name__ == '__main__':
    main()
