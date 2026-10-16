"""E1–E3 (JEDI-linear 64 p, 3-feature): units per layer, particle statements in physical units, objects = count + sums.

E1  per layer: dead units (never > 0 on real data), and each unit set to its data mean -> % of calls changed, accuracy.
E2  each l1 statement as  α_pT·pT + α_η·Δη + α_φ·Δφ + β > 0  (GeV, rad), with the spread each term contributes over particles.
E3  identity  Σ_i ReLU(w·x_i + b) = b·|A| + w·Σ_{i∈A} x_i   (A = particles passing the cut) checked on the network's own
    pooled layer P (fixed-point rounding included), and the variance of each pooled object split into count / pT / Δη / Δφ parts.
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp.jedi_core import Jedi, sample, STAGES, WIDTH

OUT = Path(__import__('os').environ.get('AI_RESULTS', 'results')) / 'jedi'; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
x, y, Q = sample(per_class=int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
net = Jedi(64)
R = net.run(x, keep=STAGES + ('logits',))
call = R['logits'].argmax(1); acc = (call == y).mean()
real = x[..., 0] > 0
print(f'{len(x)} jets, real particles per jet {real.sum(1).mean():.1f} (min {real.sum(1).min()}), accuracy {acc:.4f}', flush=True)
res = dict(N=len(x), acc=float(acc), E1={}, E2=[], E3={})

# ---------------- E1
for st in STAGES:
    v = R[st]
    flat = v[real] if v.ndim == 3 else v
    on = (flat > 0).mean(0) if st != 'z' else (flat > 0).mean(0)
    mean = flat.mean(0)
    dead = int(((flat > 0).sum(0) == 0).sum()) if st != 'z' else 0
    const = int((flat.std(0) < 1e-6).sum())
    changed = []
    for k in range(WIDTH[st]):
        m = np.zeros(WIDTH[st]); m[k] = 1
        c2 = net.run(x, override={st: (m, mean)})['logits'].argmax(1)
        changed.append(float((c2 != call).mean()))
    changed = np.array(changed)
    allm = net.run(x, override={st: (np.ones(WIDTH[st]), mean)})['logits'].argmax(1)
    res['E1'][st] = dict(dead=dead, constant=const, on_frac=on.round(4).tolist(), changed=changed.round(5).tolist(),
                         n_matter_1pct=int((changed > 0.01).sum()), n_matter_01pct=int((changed > 0.001).sum()),
                         all_mean_acc=float((allm == y).mean()))
    top = np.argsort(-changed)[:6]
    print(f'E1 {st:3s}: dead {dead:2d}/{WIDTH[st]}, constant {const:2d}, units changing >1% of calls {int((changed>0.01).sum()):2d}, '
          f'>0.1% {int((changed>0.001).sum()):2d}; top {", ".join(f"{k}:{changed[k]*100:.1f}%" for k in top)}', flush=True)

# ---------------- E2
from jetdistill.data import inputs
import json as _j
st_ = _j.loads((Path(__import__('jetdistill.config', fromlist=['DATA']).DATA) / 'norm_stats.json').read_text())['64']
shift, scale = np.array(st_['shift']), np.array(st_['scale'])
K1, b1 = net.KB['l1']                                   # (3, 64), (64,)
alpha = K1 / scale[:, None]; beta = b1 - (K1 * (shift / scale)[:, None]).sum(0)
xr = x[real].astype(np.float64)
for c in range(64):
    terms = xr * alpha[:, c]
    sd = terms.std(0); f = (xr @ alpha[:, c] + beta[c] > 0).mean()
    kind = ['pT', 'Δη', 'Δφ'][int(np.argmax(sd))]
    res['E2'].append(dict(c=c, a_pt=float(alpha[0, c]), a_eta=float(alpha[1, c]), a_phi=float(alpha[2, c]), beta=float(beta[c]),
                          sd_pt=float(sd[0]), sd_eta=float(sd[1]), sd_phi=float(sd[2]), pass_frac=float(f), main=kind,
                          pt_thr_at_axis=float(-beta[c] / alpha[0, c]) if alpha[0, c] != 0 else None))
order = np.argsort([-np.mean(np.array(res['E1']['a']['changed'])[c]) for c in range(64)])
print('E2 particle statements (top 12 by E1 cost):  α_pT [/GeV]  α_η  α_φ [/rad]  β | spread of each term | pass frac')
for c in order[:12]:
    e = res['E2'][c]
    print(f'  a{c:<2d} {e["a_pt"]:+.4f}·pT {e["a_eta"]:+7.2f}·Δη {e["a_phi"]:+7.2f}·Δφ {e["beta"]:+6.2f} > 0 | '
          f'{e["sd_pt"]:.2f} {e["sd_eta"]:.2f} {e["sd_phi"]:.2f} | {e["pass_frac"]:.2f} ({e["main"]}); cost {res["E1"]["a"]["changed"][c]*100:.1f}%')
kinds = {k: sum(1 for e in res['E2'] if e['main'] == k and res['E1']['a']['on_frac'][e['c']] > 0) for k in ('pT', 'Δη', 'Δφ')}
print('E2 dominant term among live statements:', kinds, flush=True)

# ---------------- E3
xn = inputs(x, 64).astype(np.float64)                     # the network's normalised input
A = np.einsum('jnk,kc->jnc', xn, K1) + b1 > 0               # passing set per cut (padded slots included: they are inputs too)
cnt = A.sum(1)                                            # (J, 64) particles passing each cut
sums = np.einsum('jnc,jnk->jck', A.astype(np.float64), xn)  # (J, 64, 3) summed normalised inputs of the passing particles
ident = (b1 * cnt + np.einsum('jck,kc->jc', sums, K1)) / 64
P = R['P']
err = np.abs(ident - P); rel = err.max() / np.abs(P).max()
print(f'E3 identity: max |b|A| + w·Σ_A x − P| = {err.max():.4g} (P range {np.abs(P).max():.3g}; rel {rel:.2g}); mean {err.mean():.3g}')
# variance split in physical units: count·(β) + α_pT Σ_A pT + α_η Σ_A Δη + α_φ Σ_A Δφ
xp = x.astype(np.float64)
sp = np.einsum('jnc,jnk->jck', A.astype(np.float64), xp)
parts = np.stack([beta * cnt, alpha[0] * sp[..., 0], alpha[1] * sp[..., 1], alpha[2] * sp[..., 2]], -1) / 64  # (J, 64, 4)
tot = parts.sum(-1); vt = tot.var(0) + 1e-12
cov = np.array([[np.cov(parts[:, c, i], tot[:, c])[0, 1] / vt[c] for i in range(4)] for c in range(64)])  # share of variance
res['E3'] = dict(max_err=float(err.max()), rel=float(rel), share=cov.round(4).tolist(),
                 pad_frac=float((~real).mean()), cnt_mean=cnt.mean(0).round(2).tolist())
print('E3 share of each pooled object\'s variance (count, ΣpT, ΣΔη, ΣΔφ), top 12 by E1 cost of P:')
orderP = np.argsort(-np.array(res['E1']['P']['changed']))
for c in orderP[:12]:
    print(f'  P{c:<2d} cost {res["E1"]["P"]["changed"][c]*100:5.1f}%  share {cov[c,0]:+.2f} {cov[c,1]:+.2f} {cov[c,2]:+.2f} {cov[c,3]:+.2f}  '
          f'mean |A| {cnt[:, c].mean():5.1f}  cut: {res["E2"][c]["a_pt"]:+.4f}·pT {res["E2"][c]["a_eta"]:+.2f}·Δη {res["E2"][c]["a_phi"]:+.2f}·Δφ {res["E2"][c]["beta"]:+.2f}')
(OUT / 'e1_e3.json').write_text(json.dumps(res))
print(f'done {time.time()-t0:.0f}s')
