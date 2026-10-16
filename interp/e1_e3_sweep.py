"""E1–E3 on any official JEDI-linear model (claim A7 / experiment E9 prep).

  python -m interp.e1_e3_sweep --family {3-feature,3-feature-perminv,16-feature,16-feature-perminv} --n {8,16,32,64,128}
                               [--untrained] [--seed 0] [--per-class 4000] [--acc-only]

Data: the test file (the 260,000-jet validation archive, the official test set), the first n slots of each jet:
  3-feature   val_ptetaphi_f16.npy (pT, Δη, Δφ) with val_meta.h5 labels; at n = 64 this equals _jets/n64/full_test.npz (checked)
  16-feature  val_c16_f16.npy, the 16 jetConstituentList columns of the same jets in the same order (interp/prep_16f.py)
acc_full     accuracy on all 260,000 jets (the official number: data/official_acc.json, from the JEDI-linear summary files)
zero_bits    per layer, the fraction of the layer's input-quantizer entries with 0 bits (HGQ); for qs1 also the number of kept
             (> 0 bit) slots of each live u channel
E1–E3 on a class-balanced random sample (per_class jets of each class, rng seed 0; the same jets for every model):
  E1  per stage: dead units (never > 0 on a real particle / jet; z is pre-ReLU, 0 by convention), live = width − dead,
      constant (range < 1e-6 on real data), on fraction; each unit set to its mean over real data -> fraction of calls
      changed (a unit whose replacement leaves every slot unchanged is 0 without a run); all units at their mean -> accuracy
  E2  each l1 unit as  Σ_k α_k x_k + β > 0  in physical units (x = raw features: GeV, rad, ...); the spread of each term over
      real particles; the pass fraction; top-3 terms by spread (all terms for 3 features)
  E3  identity  P = pool_scale · (b·|A| + w·Σ_A x)  on the network's own pooled layer P (fixed-point rounding included;
      A = slots passing the cut, padded slots included), and the same over A ∩ K (K = the slots the pooling layer's HGQ
      input quantizer keeps, > 0 bits; the 0-bit entries are fixed zeros); the share of each pooled object's variance
      from the count and from each summed input over A ∩ K, in physical units
untrained: the weights redrawn as Network(untrained=True) (jetdistill.network.redraw_untrained, seed)
Writes AI_RESULTS/sweep/<family>_n<n>[_untrained].json
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, STAGES, WIDTH

FAMILIES = ['3-feature', '3-feature-perminv', '16-feature', '16-feature-perminv']
QLAYERS = ['l1', 'qs', 'l2', 'dd', 'l3', 'qs1', 'd4', 'd5', 'd6', 'd7']
UNITS = dict(px='GeV', py='GeV', pz='GeV', e='GeV', pt='GeV', erel='1', ptrel='1', eta='1', etarel='1', etarot='1',
             phi='rad', phirel='rad', phirot='rad', deltaR='1', costheta='1', costhetarel='1')
ap = argparse.ArgumentParser()
ap.add_argument('--family', required=True, choices=FAMILIES)
ap.add_argument('--n', type=int, required=True, choices=[8, 16, 32, 64, 128])
ap.add_argument('--untrained', action='store_true')
ap.add_argument('--seed', type=int, default=0, help='seed of the untrained redraw')
ap.add_argument('--per-class', type=int, default=4000)
ap.add_argument('--acc-only', action='store_true')
ap.add_argument('--batch', type=int, default=0)
args = ap.parse_args()

import h5py
from jetdistill.config import DATA
t0 = time.time(); n = args.n; f16 = args.family.startswith('16-feature')
B = args.batch or (5000 if n <= 64 else 2500)
tag = f'{args.family}_n{n}' + ('_untrained' if args.untrained else '')
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'sweep'; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(DATA / ('val_c16_f16.npy' if f16 else 'val_ptetaphi_f16.npy'), mmap_mode='r')
with h5py.File(DATA / 'val_meta.h5', 'r') as f: Y = f['label'][:].astype(int)
assert len(X) == len(Y)
PT = 5 if f16 else 0                                       # the pT column (real particles: pT > 0)
net = Jedi(n, args.family, untrained=args.untrained, seed=args.seed); L = net.L
F = len(net.feat)
off = json.loads((DATA / 'official_acc.json').read_text()).get(args.family, {}).get(str(n)) if (DATA / 'official_acc.json').exists() else None
res = dict(family=args.family, n=n, untrained=args.untrained, seed=args.seed, model=net.path.name, features=net.feat,
           pool_scale=net.pool_scale, official_acc=off)
print(f'{tag}: {net.path.name}, input {tuple(net.model.input_shape)}, pool scales {net.pool_scale}, '
      f'load {time.time() - t0:.0f}s', flush=True)


# ---------------- HGQ zero-bit fractions
def qbits(layer):
    q = layer.iq.quantizer
    for a in ('b', 'bits'):
        if hasattr(q, a):
            return np.asarray(getattr(q, a), np.float64)
    k, i, f = (np.asarray(getattr(q, a), np.float64) for a in ('k', 'i', 'f'))
    return np.maximum(k + i + f, 0)


zb = {}
for name in QLAYERS:
    b = qbits(L[name]); zb[name] = dict(frac=float((b == 0).mean()), shape=list(b.shape))
bq = qbits(L['qs1']); bq = bq.reshape((-1,) + bq.shape[-1:]) if bq.ndim >= 2 else bq[None]
kept = (np.broadcast_to(bq, (n, bq.shape[-1]) if bq.shape[0] in (1, n) else bq.shape) > 0).sum(0)   # kept slots per channel
res['zero_bits'] = zb
print('zero-bit fraction per layer input: ' + ', '.join(f'{k} {v["frac"]:.3f}' for k, v in zb.items()), flush=True)

# ---------------- accuracy on the whole test file
calls = []
for i in range(0, len(X), 20000):
    calls.append(net.run(np.asarray(X[i:i + 20000, :n]).astype(np.float32), batch=B)['logits'].argmax(1))
acc_full = float((np.concatenate(calls) == Y).mean()); res['acc_full'] = acc_full; res['N_full'] = len(Y)
print(f'acc_full {acc_full:.5f} on {len(Y)} jets (official {off if off is None else f"{off:.5f}"}; '
      f'Δ {"-" if off is None else f"{(acc_full - off) * 100:+.2f} pt"}) {time.time() - t0:.0f}s', flush=True)
if args.acc_only:
    (OUT / 'acc_check').mkdir(exist_ok=True); (OUT / 'acc_check' / f'{tag}.json').write_text(json.dumps(res))
    sys.exit(0)

# ---------------- the sample
rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(Y == c), args.per_class, replace=False) for c in range(5)]))
x = np.asarray(X[idx, :n]).astype(np.float32); y = Y[idx]; real = x[..., PT] > 0
print(f'sample {len(x)} jets, real particles per jet {real.sum(1).mean():.1f} (min {real.sum(1).min()}), {time.time() - t0:.0f}s', flush=True)

# one pass: per stage, running sums over real data and the range over every slot
S = {s: dict(pos=0, s1=0., n=0, lo=np.inf, hi=-np.inf, rlo=np.inf, rhi=-np.inf) for s in STAGES}
call, Pn = [], []
for i in range(0, len(x), B):
    R = net.run(x[i:i + B], keep=STAGES + ('logits',), batch=B); rm = real[i:i + B]
    for s in STAGES:
        v = R[s]; flat = v[rm] if v.ndim == 3 else v; allv = v.reshape(-1, v.shape[-1]); d = S[s]
        d['pos'] = d['pos'] + (flat > 0).sum(0); d['s1'] = d['s1'] + flat.sum(0); d['n'] += len(flat)
        d['lo'] = np.minimum(d['lo'], allv.min(0)); d['hi'] = np.maximum(d['hi'], allv.max(0))
        d['rlo'] = np.minimum(d['rlo'], flat.min(0)); d['rhi'] = np.maximum(d['rhi'], flat.max(0))
    call.append(R['logits'].argmax(1)); Pn.append(R['P']); del R
call = np.concatenate(call); P = np.concatenate(Pn); acc = float((call == y).mean())
res.update(N=len(x), per_class=args.per_class, acc=acc, E1={}, E2=[], E3={})
res['call_frac'] = (np.bincount(call, minlength=5) / len(call)).round(4).tolist()
print(f'sample accuracy {acc:.4f}; fraction of calls per class (g q W Z t) {res["call_frac"]}, {time.time() - t0:.0f}s', flush=True)

# ---------------- E1
for s in STAGES:
    d = S[s]; W = WIDTH[s]; mean = d['s1'] / d['n']; t1 = time.time()
    dead = int((d['pos'] == 0).sum()) if s != 'z' else 0
    const = int(((d['rhi'] - d['rlo']) < 1e-6).sum())
    m32 = mean.astype(np.float32).astype(np.float64)
    noop = (d['lo'] == m32) & (d['hi'] == m32)
    changed = np.zeros(W)
    for k in np.flatnonzero(~noop):
        m = np.zeros(W); m[k] = 1
        changed[k] = (net.run(x, override={s: (m, mean)}, batch=B)['logits'].argmax(1) != call).mean()
    allm = net.run(x, override={s: (np.ones(W), mean)}, batch=B)['logits'].argmax(1)
    res['E1'][s] = dict(width=W, dead=dead, live=W - dead, live_idx=np.flatnonzero(d['pos'] > 0).tolist(), constant=const,
                        n_run=int((~noop).sum()),
                        on_frac=(d['pos'] / d['n']).round(4).tolist(), changed=changed.round(5).tolist(),
                        n_matter_1pct=int((changed > 0.01).sum()), n_matter_01pct=int((changed > 0.001).sum()),
                        all_mean_acc=float((allm == y).mean()))
    top = np.argsort(-changed)[:6]
    print(f'E1 {s:3s}: dead {dead:2d}/{W}, constant {const:2d}, units changing >1% of calls {int((changed > 0.01).sum()):2d}, '
          f'>0.1% {int((changed > 0.001).sum()):2d}; all at mean -> acc {res["E1"][s]["all_mean_acc"]:.3f}; top '
          f'{", ".join(f"{k}:{changed[k] * 100:.1f}%" for k in top)}  ({int((~noop).sum())} runs, {time.time() - t1:.0f}s)', flush=True)
live_u = res['E1']['u']['live_idx']
res['qs1_kept_slots_live_u'] = {str(c): int(kept[c]) for c in live_u}
print(f'qs1 kept slots (of {n}) per live u channel: ' + ', '.join(f'u{c}:{kept[c]}' for c in live_u), flush=True)

# ---------------- E2: the l1 cuts in physical units
K1, b1v = net.KB['l1']; b1v = b1v.reshape(-1)                     # (F, 64), (64,)
alpha = K1 / net.scale[:, None]; beta = b1v - (K1 * (net.shift / net.scale)[:, None]).sum(0)
xr = x[real].astype(np.float64); sdx = xr.std(0); sd = np.abs(alpha) * sdx[:, None]            # (F, 64)
passf = np.zeros(64)
for i in range(0, len(xr), 200000): passf += ((xr[i:i + 200000] @ alpha + beta) > 0).sum(0)
passf /= len(xr)
for c in range(64):
    o = np.argsort(-sd[:, c])
    e = dict(c=c, live=c in res['E1']['a']['live_idx'], alpha=alpha[:, c].tolist(), beta=float(beta[c]), sd=sd[:, c].tolist(),
             pass_frac=float(passf[c]), cost_a=res['E1']['a']['changed'][c], cost_P=res['E1']['P']['changed'][c],
             top3=[[net.feat[k], float(alpha[k, c]), float(sd[k, c])] for k in o[:3]])
    if not f16: e['pt_thr_at_axis'] = float(-beta[c] / alpha[0, c]) if alpha[0, c] != 0 else None
    res['E2'].append(e)
live_a = [e for e in res['E2'] if e['live']]
print(f'E2 {len(live_a)} live l1 cuts  Σ α_k x_k + β > 0  (units: ' + ', '.join(f'{k} [{UNITS[k]}]' for k in net.feat[:F]) +
      '); by E1 cost of a:', flush=True)
for e in sorted(live_a, key=lambda e: -e['cost_a']):
    terms = ' '.join(f'{a:+.4g}·{k}' for k, a, _ in e['top3'])
    print(f'  a{e["c"]:<2d} {terms} {e["beta"]:+.3g} > 0 | spread ' + ' '.join(f'{k}:{v:.2f}' for k, _, v in e['top3']) +
          f' | pass {e["pass_frac"]:.2f} | cost a {e["cost_a"] * 100:.1f}% P {e["cost_P"] * 100:.1f}%', flush=True)

# ---------------- E3: identity on the pooled layer + variance split
ps = net.pool_scale[0]; B3 = max(500, 5000 * 64 // n // (4 if f16 else 1))
kq = qbits(L['qs']); kq = kq.reshape((-1,) + kq.shape[-1:]) if kq.ndim >= 2 else kq[None]
Kq = np.broadcast_to(kq > 0, (n, 64)).astype(np.float64)          # (n, 64) slots kept (> 0 bits) by the pooling quantizer
errs, errs_k, parts = [], [], []
for i in range(0, len(x), B3):
    xb = x[i:i + B3]; xn = net.inputs(xb, n).astype(np.float64)
    A = ((xn @ K1 + b1v) > 0).astype(np.float64)                   # (J, n, 64) slots passing each cut
    cnt = A.sum(1); At = A.transpose(0, 2, 1)
    sums = At @ xn                                                 # (J, 64, F) summed normalised inputs over A
    errs.append(np.abs(ps * (b1v * cnt + (sums * K1.T[None]).sum(-1)) - P[i:i + B3]))
    AK = A * Kq[None]; cntk = AK.sum(1); AKt = AK.transpose(0, 2, 1)   # A ∩ K: passing the cut and kept by the quantizer
    errs_k.append(np.abs(ps * (b1v * cntk + ((AKt @ xn) * K1.T[None]).sum(-1)) - P[i:i + B3]))
    sp = AKt @ xb.astype(np.float64)                               # (J, 64, F) summed raw inputs over A ∩ K
    parts.append((np.concatenate([(beta * cntk)[..., None], alpha.T[None] * sp], -1) * ps).astype(np.float32))
err = np.concatenate(errs); errk = np.concatenate(errs_k); parts = np.concatenate(parts).astype(np.float64)
Pmax = max(np.abs(P).max(), 1e-12); rel = float(err.max() / Pmax); relk = float(errk.max() / Pmax)
tot = parts.sum(-1); pc = parts - parts.mean(0); tc = tot - tot.mean(0)
share = (pc * tc[..., None]).mean(0) / (tc.var(0)[:, None] + 1e-12)          # (64, 1 + F): count, Σ x_k (over A ∩ K)
la = [e['c'] for e in live_a]
res['E3'] = dict(max_err=float(err.max()), mean_err=float(err.mean()), rel=rel, P_max=float(np.abs(P).max()),
                 max_err_live=float(err[:, la].max()) if la else 0.,
                 masked=dict(max_err=float(errk.max()), mean_err=float(errk.mean()), rel=relk,
                             max_err_live=float(errk[:, la].max()) if la else 0.,
                             kept_slots=Kq.sum(0).astype(int).tolist(), zero_bit_frac=float((kq == 0).mean())),
                 share_over='A ∩ K (slots passing the cut and kept by the qs quantizer)',
                 share_cols=['count'] + net.feat, share=share.round(4).tolist(), pad_frac=float((~real).mean()))
print(f'E3 identity over all slots A: max |pool·(b|A| + w·Σ_A x) − P| = {err.max():.4g} (P range {np.abs(P).max():.3g}; '
      f'rel {rel:.2g}); mean {err.mean():.3g}', flush=True)
print(f'E3 identity over A ∩ K (K = slots with > 0 bits in the qs quantizer, {int(Kq.sum())}/{n * 64} kept): '
      f'max err {errk.max():.4g} (rel {relk:.2g}); mean {errk.mean():.3g}; live channels max {res["E3"]["masked"]["max_err_live"]:.4g}', flush=True)
print('kept slots of the live l1 channels in qs: ' + ', '.join(f'P{c}:{int(Kq[:, c].sum())}' for c in la), flush=True)
print('E3 variance share of the live pooled objects (count + top-3 summed inputs), by E1 cost of P:', flush=True)
for e in sorted(live_a, key=lambda e: -e['cost_P'])[:16]:
    c = e['c']; o = np.argsort(-np.abs(share[c, 1:]))[:3]
    print(f'  P{c:<2d} cost {e["cost_P"] * 100:5.1f}%  count {share[c, 0]:+.2f}  ' +
          '  '.join(f'Σ{net.feat[k]} {share[c, 1 + k]:+.2f}' for k in o), flush=True)
res['time_s'] = time.time() - t0
import resource
res['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024            # Linux: KiB
print(f'python peak RSS {res["peak_rss_mib"]:.0f} MiB', flush=True)
(OUT / f'{tag}.json').write_text(json.dumps(res))
print(f'wrote {OUT / f"{tag}.json"}; done {time.time() - t0:.0f}s', flush=True)
