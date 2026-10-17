"""X2: the main JEDI-linear tests run on the small taggers trained on the same inputs (train/models.py; X0/X1).

Models (N = 64 particles, best.pt of seeds 0 and 1, /j-jepa-vol/aaron-interp/models/small/<model>-n64<tag>/seed<k>/):
  dsets-d64        Deep Sets: phi = Linear(3,64) ReLU Linear(64,64) ReLU per particle; masked sum; rho = 64-64-32-16 ReLU; Linear(16,5)
  linf-2b-d32-k8   Linformer, 2 pre-LN blocks, width 32, K/V projected from 64 slots to 8 by one learned (64 x 8) matrix
  tf-2b-d32        plain pre-LN transformer, 2 blocks, width 32 (also --tag -e100: the 100-epoch runs)
Inputs exactly as in training (train/train_small.py): the first 64 particles (pT, deta, dphi), (x - shift) / scale with the
shift / scale stored in arch.json (= norm_stats.json['64']), padded rows (raw pT == 0) set to 0, mask = raw pT > 0.
Sample: interp.jedi_core.sample(per_class=4000, seed=0) = 20,000 test jets (the val archive, = the training code's test set),
the same jets as E10; fixed 50/50 split rng(0) for every read-out (= E10's split).

Layers ("units"):
  dsets  per particle: phi1 (64, ReLU), phi2 (64, ReLU); pooled: S = sum_i phi2_i (64), h1 (64), h2 (32), h3 (16, ReLU) -> logits
  tf/linf per token: emb (Linear 3->D), b<k>.attn (attention output after proj, D), b<k>.mlp (GELU hidden, 2D),
         b<k>.res (residual stream after block k, D); pooled: pool (masked mean of the final LayerNorm output, D) -> logits
Tests:
  1 accuracy on the sample vs test.json (balanced expectation = mean of test.json's per-class accuracies; binomial s.e. with the
    finite-population factor); per-jet agreement with test_probs.npy at the same rows.
  2 every unit mean-ablated (value replaced by its mean over real particles / tokens, or over jets for pooled layers): % calls
    changed vs the unedited calls. Dead = pre-activation never > 0 on real particles / jets (ReLU, GELU); constant = s.d. < 1e-6.
    Live = cost >= 1 % and >= 0.1 %. Also every unit of the layer at its mean at once (accuracy).
  3 Deep Sets only: phi1 unit k on particle i is ReLU(a_pT pT + a_eta deta + a_phi dphi + b) in GeV and rad
    (a = W1 / scale, b = b1 - sum a * shift). Summed over the real particles:
        sum_i ReLU(.) = b |A| + a_pT sum_A pT + a_eta sum_A deta + a_phi sum_A dphi     (A = real particles passing the cut)
    checked against the network's phi1 summed over real particles; variance share of each term (as E3). Each live unit
    (cost >= 0.1 %) typed by the share of its pre-activation variance over real particles: pT ramp (pT term >= 0.9), angular
    (angle terms >= 0.9), mixed; pT threshold at the axis -b / a_pT; for the angle part the normal direction and the signed
    offset of the line from the axis, c(pT) = -(b + a_pT pT) / |a_ang|, at pT = 0 and at the median real-particle pT.
  4 mass shift: every real particle moved by delta in {0, .01, .02, .03, .045, .06, .09} rad along deta and (separately) dphi;
    padding untouched (stays masked), axis fixed; invariant m unchanged (four-vector check). Positive control: deta, dphi x 1.1
    (m x 1.1, M2 x 1.21). Metrics: calls changed, W -> Z (true W called Z), Z -> W; a ridge read-out of log m (as E6v: alpha
    chosen on the last quarter of the train half, refit on the train half; fitted on unedited jets; held-out R^2 on the test
    half) at every pooled layer, the last hidden layer (h3 / pool) being the headline; its mean shift on the test half vs the
    sqrt(M0 M2) prediction 1/2 dlog(M0 M2) from the edited particles and vs 0 (invariant mass); per-jet slope; log-log exponent.
  5 variables over layers (E10 light): 12 observables (sum_pt, n_particles, sum pT r^2, pt_0, girth, mass, e2, C2, D2, tau21,
    tau32, sd_mass); held-out linear R^2 (E10's Probe: RidgeCV on standardised live units, target clipped to the train half's
    0.5-99.5 %) at each pooled layer. dsets: sum_i phi1_i ("Sa1", not a network stage: the network sums phi2), S, h1, h2, h3.
    tf/linf: each per-token layer read through [masked mean, masked sum] over the real tokens; pool as is.
Run (cluster, conda env: torch 1.13, scikit-learn 1.3, matplotlib 3.7):
  AI_PY=conda ./job.sh x2-<model> 'python -m interp.x2_small_models --model <model> [--tag -e100]'
      -> $AI_RESULTS/small/x2_<model>-n64<tag>.json
  AI_PY=conda ./job.sh x2-plot 'python -m interp.x2_small_models --plot'
      -> $AI_RESULTS/site/T8_small_mass_shift.png, T9_small_layers.png (JEDI numbers from $AI_RESULTS/jedi/e1_e3.json,
         e6_verify.json, e6_mass.json, e10_layers.json)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SMALL = Path(os.environ.get('AI_SMALL_MODELS', '/j-jepa-vol/aaron-interp/models/small'))
RES = Path(os.environ.get('AI_RESULTS', ROOT / 'results'))
OUT = RES / 'small'
SITE = RES / 'site'
CL = ['g', 'q', 'W', 'Z', 't']; W_, Z_, T_ = 2, 3, 4
DOSE = (0.01, 0.02, 0.03, 0.045, 0.06, 0.09)
OBS12 = [('sum_pt', 0, 'Σ pT'), ('n_particles', 0, 'n particles'), ('sum_ptr2', 0, 'Σ pT r²'), ('pt_0', 0, 'pT of slot 0'),
         ('girth', 1, 'girth'), ('mass', 1, 'mass'), ('e2', 1, 'e2'), ('C2', 2, 'C2'), ('D2', 2, 'D2'),
         ('tau21', 3, 'τ21'), ('tau32', 3, 'τ32'), ('sd_mass', 3, 'soft-drop mass')]
ALPHAS = 10.0 ** np.arange(-4, 4.1, 0.5)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ============================================================================================================ helpers
def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-300))


class Ridge:
    """E6v read-out: standardised inputs, alpha chosen on idx_val after a fit on idx_fit, refit on idx_all"""
    def __init__(self, X, t, idx_fit, idx_val, idx_all):
        self.mu = X[idx_all].mean(0); self.sd = X[idx_all].std(0); self.sd[self.sd < 1e-6] = np.inf
        Zs = (X - self.mu) / self.sd; self.tm = t[idx_all].mean(); self.ts = t[idx_all].std(); tt = (t - self.tm) / self.ts
        best = None
        for a in ALPHAS:
            w = self._fit(Zs[idx_fit], tt[idx_fit], a); e = ((Zs[idx_val] @ w - tt[idx_val]) ** 2).mean()
            if best is None or e < best[0]: best = (e, a)
        self.alpha = float(best[1]); self.w = self._fit(Zs[idx_all], tt[idx_all], self.alpha)

    @staticmethod
    def _fit(Zs, t, a):
        Zc = Zs - Zs.mean(0); return np.linalg.solve(Zc.T @ Zc + a * len(Zs) * np.eye(Zs.shape[1]), Zc.T @ (t - t.mean()))

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.w * self.ts + self.tm


class Probe:
    """E10 read-out: standardised inputs -> RidgeCV (alphas 1e-3..1e3), target clipped to the train half's 0.5-99.5 %"""
    def __init__(self, X, t):
        from sklearn.linear_model import RidgeCV
        self.lo, self.hi = np.percentile(t, [0.5, 99.5]); t = np.clip(t, self.lo, self.hi)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X - self.mu) / self.sd, t)

    def __call__(self, X):
        return self.m.predict((X - self.mu) / self.sd)


def moments(x):
    x = np.asarray(x, np.float64); pt, eta, phi = x[..., 0], x[..., 1], x[..., 2]
    r2_ = eta ** 2 + phi ** 2
    p4 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)], -1).sum(1)
    m = np.sqrt(np.maximum(p4[:, 3] ** 2 - (p4[:, :3] ** 2).sum(1), 0.0))
    return dict(M0=pt.sum(1), M2=(pt * r2_).sum(1), M1e=(pt * eta).sum(1), M1p=(pt * phi).sum(1), m=m,
                logm=np.log(np.maximum(m, 1e-3)), G1=(pt * np.sqrt(r2_)).sum(1), Q2=(pt ** 2 * r2_).sum(1))


def edit(x, d_eta=0.0, d_phi=0.0, k=1.0):
    """rigid shift of the real particles by (d_eta, d_phi) and / or angles x k; padded rows (pT = 0) untouched"""
    e = x.copy(); rl = (e[..., 0] > 0).astype(np.float32)
    e[..., 1] = (e[..., 1] * np.float32(k) + np.float32(d_eta)) * rl + e[..., 1] * (1 - rl)
    e[..., 2] = (e[..., 2] * np.float32(k) + np.float32(d_phi)) * rl + e[..., 2] * (1 - rl)
    return e


def rotate(x, th=0.0, reflect_phi=False):
    """rotation of the real particles about the axis by th (rad), optionally preceded by dphi -> -dphi; padding untouched"""
    e = x.copy(); rl = e[..., 0] > 0; c, s_ = np.float32(np.cos(th)), np.float32(np.sin(th))
    de, dp = x[..., 1].copy(), (-x[..., 2] if reflect_phi else x[..., 2]).copy()
    e[..., 1] = np.where(rl, c * de - s_ * dp, x[..., 1]); e[..., 2] = np.where(rl, s_ * de + c * dp, x[..., 2])
    return e


BOOT = 1000


def boot_idx(n, seed, B=BOOT):
    return np.random.default_rng(seed).integers(0, n, size=(B, n)).astype(np.int32)


def ci95(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return [float(lo), float(hi)]


def mean_ci(a, bi):
    b = a[bi].mean(1); return dict(mean=float(a.mean()), ci=ci95(b), se=float(b.std()))


def wilson(k, n, z=1.959964):
    if n == 0: return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def mcnemar(before, after):
    """exact McNemar (two-sided binomial on the discordant pairs) for paired booleans"""
    from scipy.stats import binomtest
    b = int((~before & after).sum()); c = int((before & ~after).sum())
    return dict(b=b, c=c, p=float(binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0)


def load_sample(per_class, seed):
    """interp.jedi_core.sample plus the row numbers into the test file (same rng calls) for the test_probs check"""
    from interp.jedi_core import sample
    from jetdistill.config import RESULTS
    x, y, Q = sample(per_class=per_class, seed=seed)
    yy = np.asarray(np.load(Path(RESULTS) / '_jets' / 'n64' / 'full_test.npz', mmap_mode='r')['y'])
    rng = np.random.default_rng(seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(yy == c), per_class, replace=False) for c in range(5)]))
    assert np.array_equal(yy[idx], y)
    O = {k: np.asarray(Q[k], np.float64) for k, _, _ in OBS12 if k in Q}
    O['sum_ptr2'] = np.asarray(Q['girth2'], np.float64) * np.asarray(Q['sum_pt'], np.float64)
    return np.asarray(x, np.float32), y, O, idx


# ============================================================================================================ networks
def load_net(run_dir, weights='best.pt'):
    import torch
    spec = importlib.util.spec_from_file_location('small_models', ROOT / 'train' / 'models.py')
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    arch = json.loads((run_dir / 'arch.json').read_text())
    assert not arch['cls_token']
    net = M.build_model(arch['spec'], arch['n_particles'], cls_token=False, final_ln=arch['final_ln'])
    net.load_state_dict(torch.load(run_dir / weights, map_location='cpu')); net.eval()
    return net, arch


def _ov(ov, name, t):
    if ov is None or name not in ov: return t
    m, v = ov[name]
    return t * (1 - m) + m * v


class DSRunner:
    def __init__(self, net):
        self.net = net; d = net.phi[0].out_features
        self.layers = [dict(name='phi1', width=d, tok=True, act='relu'), dict(name='phi2', width=d, tok=True, act='relu'),
                       dict(name='S', width=d, tok=False, act='relu'),
                       dict(name='h1', width=net.rho[0].out_features, tok=False, act='relu'),
                       dict(name='h2', width=net.rho[2].out_features, tok=False, act='relu'),
                       dict(name='h3', width=net.rho[4].out_features, tok=False, act='relu')]
        self.jet_layers = ['J_phi1', 'S', 'h1', 'h2', 'h3']; self.last = 'h3'

    def forward(self, x, m, ov=None):
        import torch.nn.functional as F
        n = self.net; R = {}; mm = m.unsqueeze(-1).to(x.dtype)
        z = n.phi[0](x); R['phi1:pre'] = z; R['phi1'] = _ov(ov, 'phi1', F.relu(z))
        z = n.phi[2](R['phi1']); R['phi2:pre'] = z; R['phi2'] = _ov(ov, 'phi2', F.relu(z))
        S = (R['phi2'] * mm).sum(1); R['S:pre'] = S; R['S'] = _ov(ov, 'S', S)
        h = R['S']
        for nm, i in (('h1', 0), ('h2', 2), ('h3', 4)):
            z = n.rho[i](h); R[nm + ':pre'] = z; h = R[nm] = _ov(ov, nm, F.relu(z))
        R['logits'] = n.head(h)
        return R

    def jet(self, R, m):
        mm = m.unsqueeze(-1).to(R['phi1'].dtype)
        return {'J_phi1': (R['phi1'] * mm).sum(1), 'S': R['S'], 'h1': R['h1'], 'h2': R['h2'], 'h3': R['h3']}


class TFRunner:
    def __init__(self, net):
        self.net = net; D = net.embed.out_features; B = len(net.blocks)
        self.layers = [dict(name='emb', width=D, tok=True, act='lin')]
        for b in range(B):
            self.layers += [dict(name=f'b{b}.attn', width=D, tok=True, act='lin'),
                            dict(name=f'b{b}.mlp', width=net.blocks[b].mlp[0].out_features, tok=True, act='gelu'),
                            dict(name=f'b{b}.res', width=D, tok=True, act='lin')]
        self.layers += [dict(name='pool', width=D, tok=False, act='lin')]
        self.tok_read = ['emb'] + sum([[f'b{b}.mlp', f'b{b}.res'] for b in range(B)], [])
        self.jet_layers = ['J_' + t for t in self.tok_read] + ['pool']; self.last = 'pool'

    def forward(self, x, m, ov=None):
        import torch.nn.functional as F
        n = self.net; R = {}
        h = R['emb'] = _ov(ov, 'emb', n.embed(x))
        for b, blk in enumerate(n.blocks):
            a = R[f'b{b}.attn'] = _ov(ov, f'b{b}.attn', blk.attn(blk.ln1(h), m))
            h = h + a
            z = blk.mlp[0](blk.ln2(h)); R[f'b{b}.mlp:pre'] = z
            g = R[f'b{b}.mlp'] = _ov(ov, f'b{b}.mlp', F.gelu(z))
            h = R[f'b{b}.res'] = _ov(ov, f'b{b}.res', h + blk.mlp[2](g))
        hf = n.final_ln(h); mm = m.unsqueeze(-1).to(hf.dtype)
        R['pool'] = _ov(ov, 'pool', (hf * mm).sum(1) / mm.sum(1).clamp(min=1.0))
        R['logits'] = n.head(R['pool'])
        return R

    def jet(self, R, m):
        import torch
        mm = m.unsqueeze(-1).to(R['emb'].dtype); cnt = mm.sum(1).clamp(min=1.0); J = {}
        for t in self.tok_read:
            s = (R[t] * mm).sum(1); J['J_' + t] = torch.cat([s / cnt, s], -1)
        J['pool'] = R['pool']
        return J


def normalise(x, arch):
    sh = np.asarray(arch['inputs']['shift'], np.float32); sc = np.asarray(arch['inputs']['scale'], np.float32)
    mask = x[..., 0] > 0
    xn = ((x - sh) / sc) * mask[..., None].astype(np.float32)
    return xn.astype(np.float32), mask


def run(runner, x, arch, ov=None, want=('logits',), batch=2000):
    """x raw particles (J, 64, 3) -> {key: float64 array} for logits and the jet-level layers in want"""
    import torch
    xn, mask = normalise(x, arch); out = {k: [] for k in want}
    if ov is not None:
        ov = {k: (torch.as_tensor(np.asarray(m_, np.float32)), torch.as_tensor(np.asarray(v_, np.float32))) for k, (m_, v_) in ov.items()}
    with torch.no_grad():
        for i in range(0, len(xn), batch):
            xb = torch.from_numpy(xn[i:i + batch]); mb = torch.from_numpy(mask[i:i + batch])
            R = runner.forward(xb, mb, ov)
            J = runner.jet(R, mb) if any(k in runner.jet_layers for k in want) else {}
            for k in want:
                out[k].append((J[k] if k in J else R[k]).double().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def unit_stats(runner, x, arch, batch=2000):
    """per layer: mean, s.d., max of the pre-activation (ReLU / GELU) over real particles / tokens (token layers) or jets"""
    import torch
    xn, mask = normalise(x, arch); acc = {}
    with torch.no_grad():
        for i in range(0, len(xn), batch):
            xb = torch.from_numpy(xn[i:i + batch]); mb = torch.from_numpy(mask[i:i + batch])
            R = runner.forward(xb, mb)
            for L in runner.layers:
                nm = L['name']; v = R[nm]; pre = R.get(nm + ':pre', v)
                if L['tok']: v, pre = v[mb], pre[mb]
                v = v.double(); a = acc.setdefault(nm, dict(s=0.0, s2=0.0, n=0, mx=None))
                a['s'] = a['s'] + v.sum(0); a['s2'] = a['s2'] + (v ** 2).sum(0); a['n'] += v.shape[0]
                pm = pre.double().max(0).values; a['mx'] = pm if a['mx'] is None else torch.maximum(a['mx'], pm)
    st = {}
    for nm, a in acc.items():
        mu = (a['s'] / a['n']).numpy(); sd = np.sqrt(np.maximum((a['s2'] / a['n']).numpy() - mu ** 2, 0.0))
        st[nm] = dict(mean=mu, sd=sd, premax=a['mx'].numpy(), n=a['n'])
    return st


# ============================================================================================================ tests
def calls(c, y, c0):
    return dict(acc=float((c == y).mean()), calls_changed=float((c != c0).mean()),
                W_to_Z=float((c[y == W_] == Z_).mean()), Z_to_W=float((c[y == Z_] == W_).mean()),
                W_to_W=float((c[y == W_] == W_).mean()), Z_to_Z=float((c[y == Z_] == Z_).mean()),
                W_to_t=float((c[y == W_] == T_).mean()), Z_to_t=float((c[y == Z_] == T_).mean()),
                W_changed=float((c[y == W_] != c0[y == W_]).mean()))


def test1(R0, y, run_dir, idx):
    c = R0['logits'].argmax(1); acc = float((c == y).mean())
    T = json.loads((run_dir / 'test.json').read_text()) if (run_dir / 'test.json').exists() else None
    out = dict(acc_sample=acc, per_class_sample={CL[k]: float((c[y == k] == k).mean()) for k in range(5)}, n=int(len(y)))
    if T is None:
        out['note'] = 'no test.json (run not finished)'; return out
    pc = np.array([T['per_class_accuracy'][k] for k in CL]); nk = np.array(T['confusion']).sum(1)
    per = np.bincount(y, minlength=5)
    var = sum((1 / 5) ** 2 * p * (1 - p) / n_ * (1 - n_ / N_) for p, n_, N_ in zip(pc, per, nk))
    out.update(acc_test_json=T['accuracy'], n_test=T['n'], balanced_expectation=float(pc.mean()), se=float(np.sqrt(var)),
               z=float((acc - pc.mean()) / np.sqrt(var)), class_counts_test=nk.tolist())
    tp = run_dir / 'test_probs.npy'
    if tp.exists():
        P = np.load(tp)[idx]; e = np.exp(R0['logits'] - R0['logits'].max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
        out.update(probs_maxabs_diff=float(np.abs(P - p).max()), argmax_agree=float((P.argmax(1) == c).mean()))
    return out


def test2(runner, x, arch, y, c0, st):
    res = {}
    for L in runner.layers:
        nm, Wd = L['name'], L['width']; mu = st[nm]['mean']; cost = np.zeros(Wd)
        for k in range(Wd):
            m_ = np.zeros(Wd); m_[k] = 1
            cost[k] = (run(runner, x, arch, ov={nm: (m_, mu)})['logits'].argmax(1) != c0).mean()
        allm = run(runner, x, arch, ov={nm: (np.ones(Wd), mu)})['logits'].argmax(1)
        dead = int((st[nm]['premax'] <= 0).sum()) if L['act'] in ('relu', 'gelu') else None
        N_ = len(c0); wl = np.array([wilson(int(round(v * N_)), N_) for v in cost])
        res[nm] = dict(width=Wd, tok=L['tok'], act=L['act'], dead=dead, constant=int((st[nm]['sd'] < 1e-6).sum()),
                       live_1pct=int((cost >= 0.01).sum()), live_01pct=int((cost >= 0.001).sum()),
                       live_1pct_ci_count=[int((wl[:, 0] >= 0.01).sum()), int((wl[:, 1] >= 0.01).sum())],
                       live_01pct_ci_count=[int((wl[:, 0] >= 0.001).sum()), int((wl[:, 1] >= 0.001).sum())],
                       max_cost=float(cost.max()), all_mean_acc=float((allm == y).mean()),
                       all_mean_changed=float((allm != c0).mean()), cost=cost.round(6).tolist(),
                       top=[[int(k), float(cost[k])] for k in np.argsort(-cost)[:8]])
        log(f'  T2 {nm:8s} width {Wd:3d} dead {dead} const {res[nm]["constant"]:2d} live>=1% {res[nm]["live_1pct"]:3d} '
            f'{res[nm]["live_1pct_ci_count"]} >=0.1% {res[nm]["live_01pct"]:3d} {res[nm]["live_01pct_ci_count"]} max {cost.max() * 100:5.1f}% all-mean acc {res[nm]["all_mean_acc"]:.3f}; top '
            + ', '.join(f'{k}:{v * 100:.1f}%' for k, v in res[nm]['top'][:5]))
    return res


def test3(net, arch, x, cost_phi1, J_phi1):
    """Deep Sets: first-layer cuts in GeV / rad and the count + sums identity over the real particles"""
    W1 = net.phi[0].weight.detach().double().numpy(); b1 = net.phi[0].bias.detach().double().numpy()  # (64, 3), (64,)
    sh = np.asarray(arch['inputs']['shift'], np.float64); sc = np.asarray(arch['inputs']['scale'], np.float64)
    al = W1 / sc[None]; be = b1 - (al * sh[None]).sum(1)
    real = x[..., 0] > 0; xr = x[real].astype(np.float64); ptmed = float(np.median(xr[:, 0]))
    sdx = xr.std(0)
    J, U = len(x), W1.shape[0]
    cnt = np.zeros((J, U)); sums = np.zeros((J, U, 3))
    for i in range(0, J, 2000):
        xb = x[i:i + 2000].astype(np.float64); rb = real[i:i + 2000]
        z = np.einsum('jnf,uf->jnu', xb, al) + be
        A = (z > 0) & rb[..., None]
        cnt[i:i + 2000] = A.sum(1); sums[i:i + 2000] = np.einsum('jnu,jnf->juf', A.astype(np.float64), xb)
    parts = np.concatenate([(be * cnt)[..., None], al[None] * sums], -1)          # (J, U, 4): count, pT, deta, dphi terms
    tot = parts.sum(-1)
    err = np.abs(tot - J_phi1); rel = float(err.max() / max(np.abs(J_phi1).max(), 1e-12))
    vt = tot.var(0) + 1e-12
    share = np.array([[np.cov(parts[:, u, i], tot[:, u])[0, 1] / vt[u] for i in range(4)] for u in range(U)])
    units_all = []
    for u in range(U):
        tsd = np.abs(al[u]) * sdx; v = tsd ** 2; sh_pt = float(v[0] / v.sum()); sh_ang = float(v[1:].sum() / v.sum())
        zr = xr @ al[u] + be[u]; pf = float((zr > 0).mean())
        aang = float(np.hypot(al[u, 1], al[u, 2]))
        kind = 'pT ramp' if sh_pt >= 0.9 else ('angular' if sh_ang >= 0.9 else 'mixed')
        e = dict(unit=int(u), cost=float(cost_phi1[u]) if cost_phi1 is not None else None, a_pt=float(al[u, 0]), a_eta=float(al[u, 1]), a_phi=float(al[u, 2]),
                 b=float(be[u]), sd_terms=tsd.tolist(), share_pt=sh_pt, share_ang=sh_ang, kind=kind, pass_frac=pf,
                 pt_threshold_at_axis=float(-be[u] / al[u, 0]) if al[u, 0] != 0 else None,
                 on_side='pT above' if al[u, 0] > 0 else 'pT below',
                 normal_deg=float(np.degrees(np.arctan2(al[u, 2], al[u, 1]))),
                 offset_pt0=float(-be[u] / aang) if aang > 0 else None,
                 offset_ptmed=float(-(be[u] + al[u, 0] * ptmed) / aang) if aang > 0 else None,
                 offset_per_GeV=float(-al[u, 0] / aang) if aang > 0 else None,
                 identity_share=dict(count=float(share[u, 0]), sum_pt=float(share[u, 1]), sum_deta=float(share[u, 2]),
                                     sum_dphi=float(share[u, 3])))
        units_all.append(e)
    kinds_all = {k: sum(1 for e in units_all if e['kind'] == k) for k in ('pT ramp', 'angular', 'mixed')}
    if cost_phi1 is None:
        units = units_all
    else:
        units = sorted([e for e in units_all if e['cost'] >= 0.001], key=lambda e: -e['cost'])
    kinds = {k: sum(1 for e in units if e['kind'] == k) for k in ('pT ramp', 'angular', 'mixed')}
    kinds1 = {k: sum(1 for e in units if e['kind'] == k and (e['cost'] is None or e['cost'] >= 0.01)) for k in ('pT ramp', 'angular', 'mixed')}
    ang = [e for e in units if e['kind'] == 'angular']
    through = dict(angular_offset_le_0p01=sum(1 for e in ang if abs(e['offset_ptmed']) <= 0.01),
                   angular_offset_le_0p03=sum(1 for e in ang if abs(e['offset_ptmed']) <= 0.03),
                   all_offset_le_0p01=sum(1 for e in units if e['offset_ptmed'] is not None and abs(e['offset_ptmed']) <= 0.01),
                   share_pt_quartiles=np.percentile([e['share_pt'] for e in units], [0, 25, 50, 75, 100]).tolist() if units else None,
                   always_on_pass_ge_0p95=sum(1 for e in units if e['pass_frac'] >= 0.95))
    log(f'  T3 identity sum_i phi1 = b|A| + a.sum_A x: max abs err {err.max():.3g} (rel {rel:.2g}); median real pT {ptmed:.2f} GeV; '
        f'all 64 kinds {kinds_all}; live (>=0.1%) kinds {kinds}; (>=1%) {kinds1}; {through}')
    for e in units:
        log(f'    a{e["unit"]:<2d} cost {(e["cost"] or 0) * 100:5.2f}% {e["kind"]:8s} {e["a_pt"]:+.4f}/GeV·pT {e["a_eta"]:+7.2f}·Δη '
            f'{e["a_phi"]:+7.2f}·Δφ {e["b"]:+6.2f} > 0 | var share pT {e["share_pt"]:.2f} ang {e["share_ang"]:.2f} | pass {e["pass_frac"]:.2f} '
            f'| pT* {e["pt_threshold_at_axis"] if e["pt_threshold_at_axis"] is None else round(e["pt_threshold_at_axis"], 2)} ({e["on_side"]}) '
            f'| normal {e["normal_deg"]:+6.1f}° offset(pT=0) {e["offset_pt0"]:+.3f} (pT=med) {e["offset_ptmed"]:+.3f} rad ({e["offset_per_GeV"]:+.4f}/GeV) '
            f'| sum shares cnt {e["identity_share"]["count"]:+.2f} pT {e["identity_share"]["sum_pt"]:+.2f} '
            f'η {e["identity_share"]["sum_deta"]:+.2f} φ {e["identity_share"]["sum_dphi"]:+.2f}')
    return dict(identity_max_abs_err=float(err.max()), identity_rel=rel, median_real_pt=ptmed, sd_inputs=sdx.tolist(),
                kinds_all=kinds_all, kinds_01pct=kinds, kinds_1pct=kinds1, geometry=through, units=units,
                rule='kind by share of pre-activation variance over real particles: pT ramp if pT term >= 0.9, angular if '
                     'angle terms >= 0.9, else mixed; offset c = -(b + a_pT pT)/|a_ang| (rad), c < 0: the axis is on the on side')


CAL_LOGK = (-0.2, -0.1, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)
HYP = ('sqrtM0M2', 'G1', 'sqrtQ2', 'invariant')


def test4(runner, x, arch, y, c0, R0, M, tr, te, bidx):
    """mass shift test with bootstrap intervals (over the test-half jets; class subsets for the call fractions), nulls
    (rotation by 90 deg, reflection dphi -> -dphi: r, m and M2 unchanged, prediction 0 under every hypothesis), a calibration
    curve of the read-out (deta, dphi x k: a real change of log m and of 1/2 log(M0 M2) by log k) and, per shift, the matched
    calibration edit k = exp(mean prediction)."""
    trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
    L = runner.jet_layers
    readers = {l: Ridge(R0[l], M['logm'], trf, trv, tr) for l in L}
    base = {l: readers[l](R0[l][te]) for l in L}
    tt = M['logm'][te]; sst = ((tt[bidx] - tt[bidx].mean(1, keepdims=True)) ** 2).sum(1)
    r2ci = {}
    for l in L:
        e_ = (base[l] - tt) ** 2; r2ci[l] = dict(r2=r2(base[l], tt), ci=ci95(1 - e_[bidx].sum(1) / sst))
    res = dict(readout_r2_logm={l: r2ci[l]['r2'] for l in L}, readout_r2_logm_ci={l: r2ci[l]['ci'] for l in L}, last=runner.last,
               readout_alpha={l: readers[l].alpha for l in L}, calls_unedited=calls(c0, y, c0), edits={}, n_boot=BOOT, n_test_half=int(len(te)))
    log('  T4 read-out log m held-out R² [95% CI]: ' + ', '.join(f'{l} {v["r2"]:.3f} [{v["ci"][0]:.3f}, {v["ci"][1]:.3f}]' for l, v in r2ci.items()))
    rng = np.random.default_rng(7)
    bW = rng.integers(0, int((y == W_).sum()), size=(BOOT, int((y == W_).sum())))
    bZ = rng.integers(0, int((y == Z_).sum()), size=(BOOT, int((y == Z_).sum())))
    bA = rng.integers(0, len(y), size=(BOOT, len(y))).astype(np.int32)
    p0 = dict(sqrtM0M2=0.5 * np.log(M['M0'] * M['M2']), G1=np.log(M['G1']), sqrtQ2=0.5 * np.log(M['Q2']), invariant=M['logm'])
    OBS = {}

    def one(name, xe, meta):
        Me = moments(xe); RE = run(runner, xe, arch, want=('logits',) + tuple(L)); c = RE['logits'].argmax(1)
        row = calls(c, y, c0); row.update(meta)
        isZ0, isZ1 = c0[y == W_] == Z_, c[y == W_] == Z_; isW0, isW1 = c0[y == Z_] == W_, c[y == Z_] == W_; ch = (c != c0).astype(np.float64)
        row['ci'] = dict(W_to_Z=ci95(isZ1[bW].mean(1)), Z_to_W=ci95(isW1[bZ].mean(1)), calls_changed=ci95(ch[bA].mean(1)))
        row['mcnemar'] = dict(W_to_Z=mcnemar(isZ0, isZ1), Z_to_W=mcnemar(isW0, isW1))
        pe = dict(sqrtM0M2=0.5 * np.log(Me['M0'] * Me['M2']), G1=np.log(Me['G1']), sqrtQ2=0.5 * np.log(Me['Q2']), invariant=Me['logm'])
        PR = {h: (pe[h] - p0[h])[te] for h in HYP}
        row['pred'] = {h: float(PR[h].mean()) for h in HYP}
        row['pred_sqrtM0M2'] = row['pred']['sqrtM0M2']; row['pred_sqrtM0M2_median'] = float(np.median(PR['sqrtM0M2']))
        row['dlogm_true_absmax'] = float(np.abs(Me['logm'] - M['logm'])[M['m'] > 1].max())
        row['dlogmhat'] = {}; row['dlogmhat_ci'] = {}; row['dlogmhat_se'] = {}; row['ratio'] = {}; row['ratio_ci'] = {}; row['perjet'] = {}
        OBS[name] = {}
        for l in L:
            obs = readers[l](RE[l][te]) - base[l]; OBS[name][l] = obs; mc = mean_ci(obs, bidx)
            row['dlogmhat'][l] = mc['mean']; row['dlogmhat_ci'][l] = mc['ci']; row['dlogmhat_se'][l] = mc['se']
            pp = PR['sqrtM0M2']
            if abs(pp.mean()) > 1e-6:
                row['ratio'][l] = float(obs.mean() / pp.mean()); row['ratio_ci'][l] = ci95(obs[bidx].mean(1) / pp[bidx].mean(1))
            else:
                row['ratio'][l] = None
            if pp.std() > 1e-4:
                w_ = np.linalg.lstsq(np.c_[pp, np.ones(len(pp))], obs, rcond=None)[0]
                row['perjet'][l] = dict(slope=float(w_[0]), r2=float(np.corrcoef(obs, pp)[0, 1] ** 2))
        OBS[name]['_pred'] = PR
        res['edits'][name] = row
        lt = runner.last
        log(f'  T4 {name:12s} changed {row["calls_changed"]:.3f} W->Z {row["W_to_Z"]:.3f} [{row["ci"]["W_to_Z"][0]:.3f},{row["ci"]["W_to_Z"][1]:.3f}] '
            f'(McNemar p {row["mcnemar"]["W_to_Z"]["p"]:.1e}) Z->W {row["Z_to_W"]:.3f} acc {row["acc"]:.3f} | pred ' +
            ' '.join(f'{h} {row["pred"][h]:+.4f}' for h in HYP) + f' (|Δlog m| max {row["dlogm_true_absmax"]:.1e}) | Δlog m̂ ' +
            ' '.join(f'{l} {row["dlogmhat"][l]:+.4f}' for l in L) + f' | last {row["dlogmhat"][lt]:+.4f} [{row["dlogmhat_ci"][lt][0]:+.4f},{row["dlogmhat_ci"][lt][1]:+.4f}]'
            + (f' ratio {row["ratio"][lt]:.2f} [{row["ratio_ci"][lt][0]:.2f},{row["ratio_ci"][lt][1]:.2f}]' if row['ratio'][lt] is not None else '')
            + (f' slope {row["perjet"][lt]["slope"]:.2f} r² {row["perjet"][lt]["r2"]:.2f}' if lt in row['perjet'] else ''))
        return row

    one('0', x, dict(kind='identity'))
    for ax in ('eta', 'phi'):
        for d in DOSE:
            one(f'{ax}{d}', edit(x, d if ax == 'eta' else 0.0, d if ax == 'phi' else 0.0), dict(kind='shift', axis=ax, delta=d))
    one('angle1.1', edit(x, k=1.1), dict(kind='control', angle_scale=1.1))
    one('null_rot90', rotate(x, np.pi / 2), dict(kind='null', what='rotation by 90 deg about the axis'))
    one('null_reflphi', rotate(x, 0.0, reflect_phi=True), dict(kind='null', what='reflection dphi -> -dphi'))
    for lk in CAL_LOGK:
        one(f'cal{lk}', edit(x, k=float(np.exp(lk))), dict(kind='calibration', log_k=lk))
    for ax in ('eta', 'phi'):
        for d in DOSE:
            lk = res['edits'][f'{ax}{d}']['pred_sqrtM0M2']
            one(f'match_{ax}{d}', edit(x, k=float(np.exp(lk))), dict(kind='matched calibration', axis=ax, delta=d, log_k=lk))
    # calibrated comparisons: per shift and layer, obs / matched response; each hypothesis's prediction mapped through the
    # read-out's calibration curve (log k -> response, grid incl. 0) and compared with the observed shift (paired bootstrap)
    grid = np.array(sorted((0.0,) + CAL_LOGK)); gname = ['0' if g == 0 else f'cal{g}' for g in grid]
    res['calibration_curve'] = {l: {f'{g}': res['edits'][n]['dlogmhat'][l] for g, n in zip(grid, gname)} for l in L}
    res['calibrated'] = {}
    BM = {}

    def bm(n, l):
        if (n, l) not in BM: BM[(n, l)] = OBS[n][l][bidx].mean(1)
        return BM[(n, l)]
    for ax in ('eta', 'phi'):
        for d in DOSE:
            nm = f'{ax}{d}'; out = {}
            for l in L:
                o = OBS[nm][l]; mt = OBS[f'match_{nm}'][l]; ob = bm(nm, l); mb_ = bm(f'match_{nm}', l)
                curve_b = np.stack([bm(n, l) for n in gname], 1)            # (B, grid)
                curve = np.array([OBS[n][l].mean() for n in gname])
                e = dict(obs=float(o.mean()), matched=float(mt.mean()), ratio_matched=float(o.mean() / mt.mean()) if abs(mt.mean()) > 1e-9 else None,
                         ratio_matched_ci=ci95(ob / mb_), hyp={})
                for h in HYP:
                    ph = OBS[nm]['_pred'][h]; phb = ph[bidx].mean(1)
                    rh = float(np.interp(ph.mean(), grid, curve)); rhb = np.array([np.interp(phb[i], grid, curve_b[i]) for i in range(len(phb))])
                    diff = ob - rhb; sd = float(diff.std()); z_ = float((o.mean() - rh) / sd) if sd > 0 else None
                    from scipy.stats import norm
                    e['hyp'][h] = dict(pred=float(ph.mean()), outside_grid=bool(ph.mean() > grid[-1] or ph.mean() < grid[0]),
                                       expected_response=rh, diff=float(o.mean() - rh), diff_ci=ci95(diff), z=z_,
                                       p=float(2 * norm.sf(abs(z_))) if z_ is not None else None)
                out[l] = e
            res['calibrated'][nm] = out
            lt = runner.last; e = out[lt]
            log(f'  T4 calibrated {nm:8s} {lt}: obs {e["obs"]:+.4f} matched {e["matched"]:+.4f} ratio {e["ratio_matched"]:.2f} '
                f'[{e["ratio_matched_ci"][0]:.2f},{e["ratio_matched_ci"][1]:.2f}] | ' + ' '.join(
                    f'{h}: exp {v["expected_response"]:+.4f}{"*" if v["outside_grid"] else ""} z {v["z"]:+.1f}' for h, v in e['hyp'].items()))
    ld = np.log(np.array(DOSE)); sel = np.array(DOSE) <= 0.06
    res['dose_exponent'] = {}
    for ax in ('eta', 'phi'):
        e = {}
        for l in L + ['pred']:
            v = np.array([res['edits'][f'{ax}{d}']['pred_sqrtM0M2'] if l == 'pred' else res['edits'][f'{ax}{d}']['dlogmhat'][l] for d in DOSE])[sel]
            ok = v > 0
            e[l] = float(np.polyfit(ld[sel][ok], np.log(v[ok]), 1)[0]) if ok.sum() >= 3 else None
        res['dose_exponent'][ax] = e
    log('  T4 log-log exponent (δ ≤ 0.06) η: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['dose_exponent']['eta'].items() if v is not None))
    return res


def test5(R0, O, L, tr, te, bidx):
    res = {}
    live = {l: np.flatnonzero(R0[l].std(0) > 1e-6) for l in L}
    for oid, depth, lab in OBS12:
        t = O[oid]; lo, hi = np.percentile(t[tr], [0.5, 99.5]); tc = np.clip(t, lo, hi); e = dict(depth=depth, R2={}, R2_ci={})
        tt = tc[te]; sst = ((tt[bidx] - tt[bidx].mean(1, keepdims=True)) ** 2).sum(1)
        for l in L:
            u = live[l]
            if len(u) == 0: e['R2'][l] = 0.0; e['R2_ci'][l] = [0.0, 0.0]; continue
            X = R0[l][:, u]; pr = Probe(X[tr], t[tr])(X[te]); e['R2'][l] = r2(pr, tt)
            e['R2_ci'][l] = ci95(1 - ((pr - tt) ** 2)[bidx].sum(1) / sst)
        v = [e['R2'][l] for l in L]
        e['first_0.9'] = next((i for i, q in enumerate(v) if q >= 0.9), len(L))
        e['first_0.7'] = next((i for i, q in enumerate(v) if q >= 0.7), len(L))
        res[oid] = e
        log(f'  T5 {oid:12s} d{depth} ' + ' '.join(f'{l} {e["R2"][l]:.3f} [{e["R2_ci"][l][0]:.2f},{e["R2_ci"][l][1]:.2f}]' for l in L))
    try:
        from scipy.stats import spearmanr
        d = [res[k]['depth'] for k, _, _ in OBS12]
        sp = {thr: spearmanr(d, [res[k][f'first_{thr}'] for k, _, _ in OBS12]) for thr in ('0.9', '0.7')}
        res['_spearman'] = {thr: [float(v[0]), float(v[1])] for thr, v in sp.items()}
    except Exception as ex:  # noqa: BLE001
        res['_spearman'] = str(ex)
    res['_live'] = {l: int(len(v)) for l, v in live.items()}
    return res


# ============================================================================================================ main
def main_model(a):
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    name = f'{a.model}-n{a.n}{a.tag}'
    fout = OUT / (f'x2_{name}' + ('' if a.per_class == 4000 and not a.skip_units else f'_smoke{a.per_class}') + a.part + '.json')
    log(f'X2 {name}: seeds {a.seeds}, {a.per_class} jets per class; torch {torch.__version__}, threads {torch.get_num_threads()}')
    x, y, O, idx = load_sample(a.per_class, 0); J = len(x)
    M = moments(x)
    p = np.random.default_rng(0).permutation(J); tr, te = p[:J // 2], p[J // 2:]
    bidx = boot_idx(len(te), seed=11)
    log(f'sample {J} jets, real particles per jet {(x[..., 0] > 0).sum(1).mean():.1f}; loaded {time.time() - t0:.0f}s')
    res = dict(model=a.model, n=a.n, tag=a.tag, run=name, N=J, per_class=a.per_class, sample='interp.jedi_core.sample(per_class, seed=0)',
               split='rng(0) permutation, first half train (= E10)', seeds={})
    for s in a.seeds:
        run_dir = SMALL / name / f'seed{s}'
        net, arch = load_net(run_dir)
        runner = DSRunner(net) if arch['spec']['arch'] == 'dsets' else TFRunner(net)
        log(f'seed {s}: {run_dir}; {arch["n_params"]} params; best epoch {arch.get("best", {}).get("epoch")}')
        if a.untrained_units == 'only':
            netu, _ = load_net(run_dir, 'ckpt/epoch_0.pt'); ru = DSRunner(netu) if isinstance(runner, DSRunner) else TFRunner(netu)
            R0u = run(ru, x, arch, want=('logits',) + tuple(ru.jet_layers)); c0u = R0u['logits'].argmax(1)
            stu = unit_stats(ru, x, arch); log('  untrained units:'); U = dict(test2=test2(ru, x, arch, y, c0u, stu))
            if isinstance(ru, DSRunner):
                U['test3'] = test3(netu, arch, x, np.array(U['test2']['phi1']['cost']), R0u['J_phi1'])
            res['seeds'][str(s)] = dict(untrained_units_only=True, untrained=U); fout.write_text(json.dumps(res, indent=1))
            log(f'seed {s} untrained units done [{time.time() - t0:.0f}s]'); continue
        # identity of the explicit forward with the module's own forward
        xn, mask = normalise(x[:500], arch)
        with torch.no_grad():
            ref = net(torch.from_numpy(xn), torch.from_numpy(mask)).double().numpy()
        mine = run(runner, x[:500], arch)['logits']
        fwd_diff = float(np.abs(ref - mine).max())
        R0 = run(runner, x, arch, want=('logits',) + tuple(runner.jet_layers)); c0 = R0['logits'].argmax(1)
        S = dict(forward_identity_maxdiff=fwd_diff, arch=arch['spec'], n_params=arch['n_params'], best=arch.get('best'))
        S['test1'] = test1(R0, y, run_dir, idx)
        log(f'  T1 explicit forward vs module max |Δlogit| {fwd_diff:.2e}; acc {S["test1"]["acc_sample"]:.4f}'
            + (f' vs test.json {S["test1"]["acc_test_json"]:.4f} (balanced expectation {S["test1"]["balanced_expectation"]:.4f} ± '
               f'{S["test1"]["se"]:.4f}, z {S["test1"]["z"]:+.2f}); test_probs max |Δp| {S["test1"].get("probs_maxabs_diff", float("nan")):.2e}, '
               f'argmax agree {S["test1"].get("argmax_agree", float("nan")):.4f}' if 'acc_test_json' in S['test1'] else ' (no test.json)')
            + f'  [{time.time() - t0:.0f}s]')
        S['test4'] = test4(runner, x, arch, y, c0, R0, M, tr, te, bidx); log(f'  [{time.time() - t0:.0f}s]')
        S['test5'] = test5(R0, O, runner.jet_layers, tr, te, bidx)
        log(f'  T5 Spearman depth vs first layer: {S["test5"]["_spearman"]}; live {S["test5"]["_live"]}  [{time.time() - t0:.0f}s]')
        st = unit_stats(runner, x, arch)
        if not a.skip_units:
            S['test2'] = test2(runner, x, arch, y, c0, st); log(f'  [{time.time() - t0:.0f}s]')
            if isinstance(runner, DSRunner):
                S['test3'] = test3(net, arch, x, np.array(S['test2']['phi1']['cost']), R0['J_phi1'])
        # untrained control: the same architecture at its initialisation (ckpt/epoch_0.pt of this seed)
        netu, _ = load_net(run_dir, 'ckpt/epoch_0.pt'); ru = DSRunner(netu) if isinstance(runner, DSRunner) else TFRunner(netu)
        R0u = run(ru, x, arch, want=('logits',) + tuple(ru.jet_layers)); c0u = R0u['logits'].argmax(1)
        trf, trv = tr[: len(tr) * 3 // 4], tr[len(tr) * 3 // 4:]
        U = dict(weights='ckpt/epoch_0.pt', acc=float((c0u == y).mean()), class_fractions=np.bincount(c0u, minlength=5).tolist(),
                 readout_r2_logm={l: r2(Ridge(R0u[l], M['logm'], trf, trv, tr)(R0u[l][te]), M['logm'][te]) for l in ru.jet_layers})
        log(f'  untrained: acc {U["acc"]:.4f}, calls per class {U["class_fractions"]}; read-out log m R² '
            + ', '.join(f'{l} {v:.3f}' for l, v in U['readout_r2_logm'].items()))
        U['test5'] = test5(R0u, O, ru.jet_layers, tr, te, bidx)
        if not a.skip_units and a.untrained_units == 'first' and s == a.seeds[0]:
            stu = unit_stats(ru, x, arch); log('  untrained units:')
            U['test2'] = test2(ru, x, arch, y, c0u, stu)
            if isinstance(ru, DSRunner):
                U['test3'] = test3(netu, arch, x, np.array(U['test2']['phi1']['cost']), R0u['J_phi1'])
        S['untrained'] = U
        res['seeds'][str(s)] = S
        fout.write_text(json.dumps(res, indent=1))
        log(f'seed {s} done [{time.time() - t0:.0f}s]')
    res['elapsed_s'] = float(time.time() - t0)
    fout.write_text(json.dumps(res, indent=1))
    log(f'wrote {fout} ({time.time() - t0:.0f}s)')


def load_x2(r, suffix=''):
    """x2_<run><suffix>.json, merged with any part files x2_<run>__*.json (per-seed jobs; untrained-unit-only jobs)"""
    f = OUT / f'x2_{r}{suffix}.json'
    X = json.loads(f.read_text()) if f.exists() else None
    parts = [json.loads(q.read_text()) for q in sorted(OUT.glob(f'x2_{r}__*.json'))] if not suffix else []
    for P_ in [q for q in parts if not any(S.get('untrained_units_only') for S in q['seeds'].values())]:
        if X is None: X = dict(P_, seeds={})
        X['seeds'].update(P_['seeds'])
    for P_ in [q for q in parts if any(S.get('untrained_units_only') for S in q['seeds'].values())]:
        if X is None: continue
        for sd, S in P_['seeds'].items():
            if sd in X['seeds']: X['seeds'][sd].setdefault('untrained', {}).update(S['untrained'])
    if X is not None: X['seeds'] = dict(sorted(X['seeds'].items()))
    return X


# ============================================================================================================ plot
LABEL = {'J_phi1': 'Σ phi1', 'S': 'S', 'h1': 'h1', 'h2': 'h2', 'h3': 'h3', 'J_emb': 'emb', 'J_b0.mlp': 'b0 MLP',
         'J_b0.res': 'b0', 'J_b1.mlp': 'b1 MLP', 'J_b1.res': 'b1', 'pool': 'pool'}
SHORT = {'dsets-d64-n64': 'Deep Sets', 'linf-2b-d32-k8-n64': 'Linformer', 'tf-2b-d32-n64': 'Transformer',
         'tf-2b-d32-n64-e100': 'Transformer 100 ep'}


def _table(fname, title, cols, rows, widths, note=None, fs=8.0, line_in=0.15, pad_in=0.08, shade=None, center_from=1, vlines=()):
    """matplotlib table whose row heights follow the number of text lines in each row (cells wrap only at explicit newlines)"""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    allr = [cols] + rows
    lines = [max(str(c).count('\n') + 1 for c in r_) for r_ in allr]
    hin = sum(n_ * line_in + pad_in for n_ in lines)
    title_in = 0.42; note_in = (note.count('\n') + 1) * 0.16 + 0.12 if note else 0.05
    H = hin + title_in + note_in
    fig = plt.figure(figsize=(sum(widths), H), dpi=160)
    ax = fig.add_axes([0.004, note_in / H, 0.992, hin / H]); ax.axis('off')
    t = ax.table(cellText=rows, colLabels=cols, colWidths=[w / sum(widths) for w in widths], bbox=[0, 0, 1, 1],
                 cellLoc='left', colLoc='left')
    t.auto_set_font_size(False); t.set_fontsize(fs)
    for (r, c), cell in t.get_celld().items():
        cell.set_height((lines[r] * line_in + pad_in) / hin); cell.set_edgecolor('#d8d4ca'); cell.PAD = 0.04
        if c >= center_from: cell.set_text_props(ha='center')
        if r == 0: cell.set_facecolor('#eeebe3'); cell.set_text_props(weight='bold')
        elif shade is not None and shade(r - 1, c) is not None: cell.set_facecolor(shade(r - 1, c))
        elif r % 2 == 0: cell.set_facecolor('#f8f7f3')
    cw = np.cumsum([0.0] + [w / sum(widths) for w in widths])
    for j in vlines:
        ax.plot([cw[j], cw[j]], [0, 1], transform=ax.transAxes, color='#6f6a5f', lw=1.4, clip_on=False, zorder=5)
    fig.text(0.004, 1 - 0.12 / H, title, fontsize=10, weight='bold', va='top', ha='left')
    if note: fig.text(0.004, 0.04 / H, note, fontsize=7.2, color='#555', va='bottom', ha='left', linespacing=1.35)
    SITE.mkdir(parents=True, exist_ok=True); fig.savefig(SITE / fname, bbox_inches='tight', pad_inches=0.08); print('wrote', SITE / fname)


def plot(suffix='', only=None):
    J6v = json.loads((RES / 'jedi' / 'e6_verify.json').read_text()); J6 = json.loads((RES / 'jedi' / 'e6_mass.json').read_text())
    J10 = json.loads((RES / 'jedi' / 'e10_layers.json').read_text())
    X = {r: load_x2(r, suffix) for r in ('dsets-d64-n64', 'linf-2b-d32-k8-n64', 'tf-2b-d32-n64', 'tf-2b-d32-n64-e100') if not only or r in only}
    X = {r: v for r, v in X.items() if v is not None and v['seeds']}; runs = list(X)
    pc = lambda v: f'{100 * v:.0f}' if v is not None else '–'
    f2 = lambda v: f'{v:.2f}' if v is not None else '–'
    # ---------------------------------------------------------------- T8
    cols = ['model (seed)\nread-out layer', 'acc\n(sample)', 'R² log m\n[95 % CI]', 'W → Z [%]\nδ = 0 / .03 / .06 / .09',
            'Z → W [%]\nδ = 0 / .06', 'calls changed [%]\nδ = .03 / .06;\nnull rot. 90°', 'Δlog m̂ at δ = .06\n[95 % CI]\n(√(M0M2): pred.; inv. m: 0)',
            'Δlog m̂ ÷ ½Δlog(M0M2)\nδ = .01 .02 .03\nδ = .045 .06 .09', 'same, calibrated:\n÷ matched angle × k\nδ = .03 / .06 [CI at .06]',
            'φ shift, δ = .06\nraw / calibrated', 'null rot. 90°\nΔlog m̂ [CI]', 'control angles × 1.1\nΔlog m̂ ÷ log 1.1;\nW → Z [%]']
    E = J6v['edits']; u = J6v['calls_unedited']; zj = lambda k: E[k]['dlogmhat']['z'] / E[k]['pred_sqrtM0M2']
    k11 = J6['transfer']['free']['z']['k1.1'] / np.log(1.1)
    se06 = E['0.06_eta']['dlogmhat_se']['z']
    rows = [['JEDI-linear (E6v; × 1.1: E6)\nz (16 neurons)', f'{J6v["acc"]:.3f}', f'{J6v["readout_r2_logm"]["z"]:.3f}',
             ' / '.join(pc(v) for v in (u['W_to_Z'], E['0.03_eta']['W_to_Z'], E['0.06_eta']['W_to_Z'], E['0.09_eta']['W_to_Z'])),
             f'{pc(u["Z_to_W"])} / {pc(E["0.06_eta"]["Z_to_W"])}', 'n/r (W jets\nchanged ' + f'{pc(E["0.03_eta"]["W_changed"])} / {pc(E["0.06_eta"]["W_changed"])})\n4.1 (E5v)',
             f'{E["0.06_eta"]["dlogmhat"]["z"]:.3f} [±{1.96 * se06:.3f}]\n({E["0.06_eta"]["pred_sqrtM0M2"]:.3f}; 0)',
             ' '.join(f2(zj(f'{d}_eta')) for d in DOSE[:3]) + '\n' + ' '.join(f2(zj(f'{d}_eta')) for d in DOSE[3:]),
             'n/r', f'{f2(zj("0.06_phi"))} / n/r', 'n/r', f'{k11:.2f}; {pc(J6["calls"]["free"]["k1.1"]["W"]["Z"])}']]
    for r in runs:
        for s, S in X[r]['seeds'].items():
            T4 = S['test4']; e = T4['edits']; lt = T4['last']; q = lambda k: e[k]['ratio'][lt]; C = T4['calibrated']
            ci_ = T4['readout_r2_logm_ci'][lt]; d6 = e['eta0.06']['dlogmhat_ci'][lt]; nr = e['null_rot90']
            rows.append([f'{SHORT.get(r, r)} (s{s})\n{LABEL[lt]}', f'{S["test1"]["acc_sample"]:.3f}',
                         f'{T4["readout_r2_logm"][lt]:.3f}\n[{ci_[0]:.3f}, {ci_[1]:.3f}]',
                         ' / '.join(pc(e[k]['W_to_Z']) for k in ('0', 'eta0.03', 'eta0.06', 'eta0.09')),
                         f'{pc(e["0"]["Z_to_W"])} / {pc(e["eta0.06"]["Z_to_W"])}',
                         f'{pc(e["eta0.03"]["calls_changed"])} / {pc(e["eta0.06"]["calls_changed"])}\n{pc(nr["calls_changed"])}',
                         f'{e["eta0.06"]["dlogmhat"][lt]:.3f} [{d6[0]:.3f}, {d6[1]:.3f}]\n({e["eta0.06"]["pred_sqrtM0M2"]:.3f}; 0)',
                         ' '.join(f2(q(f'eta{d}')) for d in DOSE[:3]) + '\n' + ' '.join(f2(q(f'eta{d}')) for d in DOSE[3:]),
                         f'{f2(C["eta0.03"][lt]["ratio_matched"])} / {f2(C["eta0.06"][lt]["ratio_matched"])}\n'
                         f'[{C["eta0.06"][lt]["ratio_matched_ci"][0]:.2f}, {C["eta0.06"][lt]["ratio_matched_ci"][1]:.2f}]',
                         f'{f2(q("phi0.06"))} / {f2(C["phi0.06"][lt]["ratio_matched"])}',
                         f'{round(nr["dlogmhat"][lt], 3) + 0.0:+.3f}\n[{round(nr["dlogmhat_ci"][lt][0], 3) + 0.0:+.3f}, {round(nr["dlogmhat_ci"][lt][1], 3) + 0.0:+.3f}]',
                         f'{e["angle1.1"]["dlogmhat"][lt] / np.log(1.1):.2f}; {pc(e["angle1.1"]["W_to_Z"])}'])
    note = ('Edit: every real particle moved by δ rad along Δη (φ column: along Δφ), padding untouched, axis fixed; invariant mass unchanged (|Δlog m| < 1e-6). '
            'Read-out: ridge fit of log m on the last hidden layer of unedited jets (train half),\nmean shift on the test half (10,000 jets); 95 % intervals: bootstrap over jets (1,000 resamples). '
            'Prediction of the mass about the given axis √(M0·M2): ½Δlog(M0·M2) (M0 = Σ pT, M2 = Σ pT r²); of the invariant mass: 0.\n'
            'Calibrated: shift response ÷ the response to Δη, Δφ × k with log k = the mean prediction (a real change of m and of √(M0M2) by the same amount). '
            'Null: rotation by 90° about the axis (r, M2 unchanged; prediction 0).\n'
            'JEDI-linear: 10,000 test jets (E6v; ±: 1.96 s.e.), control 15,000 (E6), rotation 10,000 (E5v, fixed point); n/r = not recorded. Small models: 20,000 test jets, seeds 0 and 1.')
    _table(f'T8_small_mass_shift{suffix}.png', 'Mass shift test (rigid translation, invariant mass unchanged): JEDI-linear and the small taggers trained on the same inputs',
           cols, rows, [2.0, 0.65, 1.05, 1.55, 0.95, 1.25, 1.75, 1.6, 1.55, 1.2, 1.15, 1.35], note=note, fs=7.8)
    # ---------------------------------------------------------------- T9
    JL = ['P', 'g', 'S', 'h4', 'h5', 'z']
    groups = [('JEDI', JL, lambda oid, l: J10['obs'][oid]['R2']['trained'][l]['lin'])]
    for r in runs:
        Ls = list(next(iter(X[r]['seeds'].values()))['test5'][OBS12[0][0]]['R2'].keys())
        groups.append((SHORT.get(r, r), Ls, (lambda rr: (lambda oid, l: float(np.mean([S['test5'][oid]['R2'][l] for S in X[rr]['seeds'].values()]))))(r)))
    GH = {'JEDI': 'JEDI-\nlinear', 'Deep Sets': 'Deep\nSets', 'Linformer': 'Lin-\nformer', 'Transformer': 'Trans-\nformer',
          'Transformer 100 ep': 'Transf.\n100 ep'}
    cols = ['observable', 'depth'] + [f'{GH.get(g, g)}\n{LABEL.get(l, l)}' for g, Ls, _ in groups for l in Ls]
    rows, vals = [], []
    for oid, depth, lab in OBS12:
        v = [f(oid, l) for g, Ls, f in groups for l in Ls]; vals.append(v)
        rows.append([lab, str(depth)] + [f'{q:.2f}' for q in v])
    cnt = [sum(1 for i in range(len(OBS12)) if vals[i][j] >= 0.9) for j in range(len(vals[0]))]
    rows.append(['R² ≥ 0.9 (of 12)', '']); rows[-1] += [str(c) for c in cnt]
    ucol = [[J10['obs'][oid]['R2']['untrained'][l]['lin'] for oid, _, _ in OBS12] for l in JL]
    for r in runs:
        Ls = groups[[g for g, _, _ in groups].index(SHORT.get(r, r))][1]
        for l in Ls:
            ucol.append([float(np.mean([S['untrained']['test5'][oid]['R2'][l] for S in X[r]['seeds'].values()])) for oid, _, _ in OBS12])
    rows.append(['untrained: median R²', '']); rows[-1] += [f'{round(float(np.median(c)), 2) + 0.0:.2f}' for c in ucol]
    rows.append(['untrained: R² ≥ 0.9', '']); rows[-1] += [str(sum(1 for v in c if v >= 0.9)) for c in ucol]
    hw = max(max((S['test5'][oid]['R2_ci'][l][1] - S['test5'][oid]['R2_ci'][l][0]) / 2 for oid, _, _ in OBS12
                 for l in S['test5'][OBS12[0][0]]['R2']) for r in runs for S in X[r]['seeds'].values())
    starts = list(np.cumsum([len(Ls) for _, Ls, _ in groups])[:-1] + 2)

    def shade(i, c):
        if i >= len(OBS12) or c < 2: return None
        q = max(0.0, min(1.0, vals[i][c - 2]))
        return (1 - 0.55 * q, 1 - 0.30 * q, 1 - 0.05 * q)
    note = ('Held-out linear R² (RidgeCV on the live units, 10,000 / 10,000 jets, target clipped to 0.5–99.5 %), the 20,000 test jets of E10. '
            'Small models: mean of seeds 0 and 1; largest 95 % bootstrap half-width of a small-model cell ' + f'{hw:.3f}.' + '\nUntrained: the same architecture at '
            'initialisation (JEDI: weights redrawn, E10; small models: ckpt/epoch_0.pt). JEDI stages: P = Σ a / 64, g context, S = Σ u / 16, h4, h5, z. Deep Sets: Σ phi1 = first layer '
            'summed over particles (not a network stage; the network sums phi2), S = Σ phi2, h1–h3 dense.\nTransformers: emb, b<k> MLP (GELU hidden), '
            'b<k> (residual stream after block k) read through [mean, sum] over the real tokens; pool = the network\'s pooled vector.')
    widths = [1.25, 0.45] + [0.56] * (len(cols) - 2)
    _table(f'T9_small_layers{suffix}.png', 'Variables over layers: held-out linear R² of 12 jet observables at each layer (depth 0 = sums, 1 = products / ratios, '
           '2 = 3-point ratios, 3 = clustering)', cols, rows, widths, note=note, fs=7.5, shade=shade, center_from=1, vlines=[2] + starts)


# ============================================================================================================ report
def report(only=None, suffix=''):
    """Markdown tables of tests 1-5 from the JSONs (printed; the source of inbox/X2_small_models.md)"""
    X = {r: load_x2(r, suffix) for r in ('dsets-d64-n64', 'linf-2b-d32-k8-n64', 'tf-2b-d32-n64', 'tf-2b-d32-n64-e100') if not only or r in only}
    X = {r: v for r, v in X.items() if v is not None and v['seeds']}; runs = list(X)
    J1 = json.loads((RES / 'jedi' / 'e1_e3.json').read_text())
    P = lambda *a: print(*a)
    pc = lambda v, d=1: f'{100 * v:.{d}f}'
    # ---- summary tables (the contents of T8 and T9)
    P('## Summary: mass shift at the last hidden layer (T8 contents)\n')
    P('| run (seed), layer | R² log m [CI] | W → Z % δ = 0/.03/.06/.09 (η) | Z → W % δ = 0/.06 | calls changed % η .03/.06 | Δlog m̂ η .06 [CI] | raw ratio η .01/.02/.03/.045/.06/.09 | '
      'calibrated ratio η .03/.06/.09 [CI at .06] | φ .06 raw / calibrated | rot 90°: Δlog m̂ [CI]; calls % | × 1.1: Δlog m̂ ÷ log 1.1; W → Z % | exponent η (δ ≤ .06) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            T = S['test4']; e = T['edits']; lt = T['last']; C = T['calibrated']; ci_ = T['readout_r2_logm_ci'][lt]
            f2 = lambda v: f'{v:.2f}' if v is not None else '–'
            ex = T['dose_exponent']['eta'].get(lt)
            P(f'| {r} ({sd}), {lt} | {T["readout_r2_logm"][lt]:.3f} [{ci_[0]:.3f}, {ci_[1]:.3f}] | '
              + '/'.join(pc(e[k]['W_to_Z'], 0) for k in ('0', 'eta0.03', 'eta0.06', 'eta0.09')) + f' | {pc(e["0"]["Z_to_W"], 0)}/{pc(e["eta0.06"]["Z_to_W"], 0)} | '
              f'{pc(e["eta0.03"]["calls_changed"], 1)}/{pc(e["eta0.06"]["calls_changed"], 1)} | {e["eta0.06"]["dlogmhat"][lt]:.3f} [{e["eta0.06"]["dlogmhat_ci"][lt][0]:.3f}, {e["eta0.06"]["dlogmhat_ci"][lt][1]:.3f}] | '
              + '/'.join(f2(e[f'eta{d}']['ratio'][lt]) for d in DOSE) + ' | '
              + '/'.join(f2(C[f'eta{d}'][lt]['ratio_matched']) for d in (0.03, 0.06, 0.09)) + f' [{C["eta0.06"][lt]["ratio_matched_ci"][0]:.2f}, {C["eta0.06"][lt]["ratio_matched_ci"][1]:.2f}] | '
              f'{f2(e["phi0.06"]["ratio"][lt])} / {f2(C["phi0.06"][lt]["ratio_matched"])} | '
              f'{round(e["null_rot90"]["dlogmhat"][lt], 4) + 0.0:+.4f} [{round(e["null_rot90"]["dlogmhat_ci"][lt][0], 4) + 0.0:+.4f}, {round(e["null_rot90"]["dlogmhat_ci"][lt][1], 4) + 0.0:+.4f}]; {pc(e["null_rot90"]["calls_changed"], 1)} | '
              f'{e["angle1.1"]["dlogmhat"][lt] / np.log(1.1):.2f}; {pc(e["angle1.1"]["W_to_Z"], 1)} | {ex if ex is None else round(ex, 2)} |')
    P('\nCalibrated ratio at η + 0.06 by layer (shift response ÷ matched angle-scale response):\n')
    P('| run (seed) | ' + ' | '.join(['layer: ratio [CI]']) + ' |'); P('|---|---|')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            C = S['test4']['calibrated']['eta0.06']
            P(f'| {r} ({sd}) | ' + '; '.join(f'{LABEL.get(l, l)} {C[l]["ratio_matched"]:.2f} [{C[l]["ratio_matched_ci"][0]:.2f}, {C[l]["ratio_matched_ci"][1]:.2f}]'
                                          for l in C if not l.startswith('J_emb')) + ' |')
    P('\n## Summary: held-out linear R² by layer, mean of seeds (T9 contents)\n')
    J10 = json.loads((RES / 'jedi' / 'e10_layers.json').read_text()); JL = ['P', 'g', 'S', 'h4', 'h5', 'z']
    for r in [None] + runs:
        Ls = JL if r is None else list(next(iter(X[r]['seeds'].values()))['test5'][OBS12[0][0]]['R2'].keys())
        val = (lambda oid, l: J10['obs'][oid]['R2']['trained'][l]['lin']) if r is None else \
            (lambda oid, l, rr=r: float(np.mean([S['test5'][oid]['R2'][l] for S in X[rr]['seeds'].values()])))
        unt = (lambda oid, l: J10['obs'][oid]['R2']['untrained'][l]['lin']) if r is None else \
            (lambda oid, l, rr=r: float(np.mean([S['untrained']['test5'][oid]['R2'][l] for S in X[rr]['seeds'].values()])))
        P(f'\n{"JEDI-linear (E10)" if r is None else r}\n')
        P('| observable | depth | ' + ' | '.join(LABEL.get(l, l) for l in Ls) + ' |'); P('|---|---|' + '---|' * len(Ls))
        for oid, d, lab in OBS12:
            P(f'| {lab} | {d} | ' + ' | '.join(f'{val(oid, l):.2f}' for l in Ls) + ' |')
        P('| R² ≥ 0.9 (of 12) | | ' + ' | '.join(str(sum(val(o, l) >= 0.9 for o, _, _ in OBS12)) for l in Ls) + ' |')
        P('| untrained: median R² | | ' + ' | '.join(f'{round(float(np.median([unt(o, l) for o, _, _ in OBS12])), 2) + 0.0:.2f}' for l in Ls) + ' |')
        P('| untrained: R² ≥ 0.9 | | ' + ' | '.join(str(sum(unt(o, l) >= 0.9 for o, _, _ in OBS12)) for l in Ls) + ' |')

    P('## Test 1: accuracy on the 20,000-jet sample\n')
    P('| model | seed | acc (sample) | test.json acc (260k) | balanced expectation ± s.e. | z | max abs Δp vs test_probs | argmax agreement |')
    P('|---|---|---|---|---|---|---|---|')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            t = S['test1']
            if 'acc_test_json' in t:
                P(f'| {r} | {sd} | {t["acc_sample"]:.4f} | {t["acc_test_json"]:.4f} | {t["balanced_expectation"]:.4f} ± {t["se"]:.4f} | {t["z"]:+.2f} | '
                  f'{t.get("probs_maxabs_diff", float("nan")):.1e} | {t.get("argmax_agree", float("nan")):.4f} |')
            else:
                P(f'| {r} | {sd} | {t["acc_sample"]:.4f} | – | – | – | – | – |')
    P('\n## Test 2: units (mean ablation, % of the 20,000 calls changed)\n')
    P('Columns: width; dead (pre-activation never > 0 on real particles / tokens / jets; ReLU and GELU layers only); live ≥ 1 % and ≥ 0.1 % '
      '(in brackets: units whose 95 % Wilson interval lies entirely above the threshold, and units whose upper bound reaches it); largest single-unit cost; '
      'accuracy with every unit of the layer at its mean.\n')
    P('| model | seed | layer | width | dead | live ≥ 1 % | live ≥ 0.1 % | max cost | all-mean acc |')
    P('|---|---|---|---|---|---|---|---|---|')
    for st in ('a', 'P', 'g', 's', 'u', 'S', 'h4', 'h5', 'z'):
        e = J1['E1'][st]
        P(f'| JEDI-linear (E1, 10k jets) | – | {st} | {len(e["changed"])} | {e["dead"]} | {e["n_matter_1pct"]} | {e["n_matter_01pct"]} | {100 * max(e["changed"]):.1f} % | {e["all_mean_acc"]:.3f} |')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            for tag, T in (('', S.get('test2')), (' untrained', S.get('untrained', {}).get('test2'))):
                if not T: continue
                for nm, e in T.items():
                    P(f'| {r}{tag} | {sd} | {nm} | {e["width"]} | {e["dead"] if e["dead"] is not None else "–"} | {e["live_1pct"]} {e["live_1pct_ci_count"]} | '
                      f'{e["live_01pct"]} {e["live_01pct_ci_count"]} | {100 * e["max_cost"]:.1f} % | {e["all_mean_acc"]:.3f} |')
    P('\n## Test 3: Deep Sets first layer as cuts\n')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            for tag, T in (('trained', S.get('test3')), ('untrained', S.get('untrained', {}).get('test3'))):
                if not T: continue
                P(f'- {r} seed {sd} {tag}: identity max |Σ phi1 − (b|A| + a·Σ_A x)| = {T["identity_max_abs_err"]:.2e} (relative {T["identity_rel"]:.1e}); '
                  f'median real-particle pT {T["median_real_pt"]:.2f} GeV; kinds over all 64 units {T["kinds_all"]}; over live units (≥ 0.1 %) {T["kinds_01pct"]}; '
                  f'geometry {json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in T["geometry"].items()})}; '
                  'a_pT < 0 (on for soft particles) / > 0 among live units by kind: ' + ', '.join(
                      f'{k} {sum(1 for e in T["units"] if e["kind"] == k and e["a_pt"] < 0)} / {sum(1 for e in T["units"] if e["kind"] == k and e["a_pt"] > 0)}'
                      for k in ('pT ramp', 'angular', 'mixed')) + '; |offset| at median pT ≤ 0.01 / ≤ 0.03 / > 0.03 rad among live mixed units: ' + ' / '.join(
                      str(sum(1 for e in T['units'] if e['kind'] == 'mixed' and f(abs(e['offset_ptmed'])))) for f in
                      (lambda v: v <= 0.01, lambda v: v <= 0.03, lambda v: v > 0.03)))
    for r in runs:
        S = X[r]['seeds'].get('0'); T = S.get('test3') if S else None
        if not T: continue
        P(f'\n{r} seed 0, live first-layer units by cost (cut: a_pT·pT + a_η·Δη + a_φ·Δφ + b > 0, pT in GeV, angles in rad; '
          'share = fraction of pre-activation variance over real particles; offset c = −(b + a_pT·pT)/|a_ang| at the median particle pT, c < 0: the axis is on the on side; '
          'last column: variance shares of Σ_i phi1 from count·b, a_pT Σ_A pT, a_η Σ_A Δη, a_φ Σ_A Δφ)\n')
        P('| unit | cost | kind | a_pT [1/GeV] | a_η | a_φ | b | pT share | pass frac | pT at axis [GeV] | normal [deg] | offset c [rad] | dc/dpT [rad/GeV] | shares cnt / pT / η / φ |')
        P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for e in T['units']:
            P(f'| a{e["unit"]} | {pc(e["cost"])} % | {e["kind"]} | {e["a_pt"]:+.4f} | {e["a_eta"]:+.2f} | {e["a_phi"]:+.2f} | {e["b"]:+.2f} | {e["share_pt"]:.2f} | '
              f'{e["pass_frac"]:.2f} | {e["pt_threshold_at_axis"]:.1f} ({e["on_side"]}) | {e["normal_deg"]:+.0f} | {e["offset_ptmed"]:+.3f} | {e["offset_per_GeV"]:+.4f} | '
              f'{e["identity_share"]["count"]:+.2f} / {e["identity_share"]["sum_pt"]:+.2f} / {e["identity_share"]["sum_deta"]:+.2f} / {e["identity_share"]["sum_dphi"]:+.2f} |')
    P('\n## Test 4: mass shift\n')
    for r in runs:
        for sd, S in X[r]['seeds'].items():
            T = S['test4']; lt = T['last']; L = list(T['readout_r2_logm'].keys())
            P(f'\n### {r} seed {sd} (read-out layer {lt}; test half {T["n_test_half"]} jets, {T["n_boot"]} bootstrap resamples)\n')
            P('Read-out of log m, held-out R² [95 % CI]: ' + '; '.join(f'{l} {T["readout_r2_logm"][l]:.3f} [{T["readout_r2_logm_ci"][l][0]:.3f}, {T["readout_r2_logm_ci"][l][1]:.3f}]' for l in L)
              + f'. Untrained (epoch 0): ' + '; '.join(f'{l} {v:.3f}' for l, v in S['untrained']['readout_r2_logm'].items()) + '\n')
            P('| edit | calls changed [CI] | W → Z [CI] (McNemar p) | Z → W | pred ½Δlog(M0M2) / ΔlogG1 / ½ΔlogQ2 | Δlog m̂ (last) [CI] | ratio to ½Δlog(M0M2) [CI] | per-jet slope, r² |')
            P('|---|---|---|---|---|---|---|---|')
            for nm, e in T['edits'].items():
                if nm.startswith('cal') or nm.startswith('match'): continue
                rt = e['ratio'].get(lt); pj = e['perjet'].get(lt)
                P(f'| {nm} | {pc(e["calls_changed"])} [{pc(e["ci"]["calls_changed"][0])}, {pc(e["ci"]["calls_changed"][1])}] | '
                  f'{pc(e["W_to_Z"])} [{pc(e["ci"]["W_to_Z"][0])}, {pc(e["ci"]["W_to_Z"][1])}] ({e["mcnemar"]["W_to_Z"]["p"]:.1e}) | {pc(e["Z_to_W"])} | '
                  f'{e["pred"]["sqrtM0M2"]:.3f} / {e["pred"]["G1"]:.3f} / {e["pred"]["sqrtQ2"]:.3f} | {e["dlogmhat"][lt]:+.4f} [{e["dlogmhat_ci"][lt][0]:+.4f}, {e["dlogmhat_ci"][lt][1]:+.4f}] | '
                  + (f'{rt:.2f} [{e["ratio_ci"][lt][0]:.2f}, {e["ratio_ci"][lt][1]:.2f}]' if rt is not None else '–') + ' | '
                  + (f'{pj["slope"]:.2f}, {pj["r2"]:.2f}' if pj else '–') + ' |')
            P('\nCalibrated comparison at the read-out layer (matched edit: Δη, Δφ × k, log k = mean prediction; expected response of each hypothesis from the '
              'calibration curve, * = prediction beyond the curve\'s range 0.9; z = (observed − expected) / bootstrap s.d.):\n')
            P('| shift | observed | matched response | ratio [CI] | z vs √(M0M2) | z vs Σ pT r | z vs √Σ pT² r² | z vs invariant m (0) |')
            P('|---|---|---|---|---|---|---|---|')
            for nm, C in T['calibrated'].items():
                c = C[lt]; h = c['hyp']
                zf = lambda k: f'{h[k]["z"]:+.1f}{"*" if h[k]["outside_grid"] else ""}' if h[k]['z'] is not None else '–'
                P(f'| {nm} | {c["obs"]:+.4f} | {c["matched"]:+.4f} | {c["ratio_matched"]:.2f} [{c["ratio_matched_ci"][0]:.2f}, {c["ratio_matched_ci"][1]:.2f}] | '
                  f'{zf("sqrtM0M2")} | {zf("G1")} | {zf("sqrtQ2")} | {zf("invariant")} |')
            P('\nAll read-out layers, Δlog m̂ at η + 0.06 / φ + 0.06 / angles × 1.1 / rotation 90° (raw ratio to ½Δlog(M0M2) at η + 0.06; calibrated ratio at η + 0.06):\n')
            P('| layer | η + .06 | φ + .06 | × 1.1 | rot 90° | raw ratio η .06 | calibrated ratio η .06 [CI] | exponent η (δ ≤ .06) |')
            P('|---|---|---|---|---|---|---|---|')
            for l in L:
                e6, p6, a1, n9 = (T['edits'][k] for k in ('eta0.06', 'phi0.06', 'angle1.1', 'null_rot90')); c = T['calibrated']['eta0.06'][l]
                ex = T['dose_exponent']['eta'].get(l)
                P(f'| {l} | {e6["dlogmhat"][l]:+.3f} | {p6["dlogmhat"][l]:+.3f} | {a1["dlogmhat"][l]:+.3f} | {n9["dlogmhat"][l]:+.4f} | {e6["ratio"][l]:.2f} | '
                  f'{c["ratio_matched"]:.2f} [{c["ratio_matched_ci"][0]:.2f}, {c["ratio_matched_ci"][1]:.2f}] | {ex if ex is None else round(ex, 2)} |')
            P(f'\nPrediction exponent (½Δlog(M0M2), η): {T["dose_exponent"]["eta"]["pred"]:.2f}.')
    P('\n## Test 5: variables over layers (held-out linear R², per seed, [95 % CI]; untrained in the last column group)\n')
    try:
        from scipy.stats import spearmanr
        J10 = json.loads((RES / 'jedi' / 'e10_layers.json').read_text())
        f9 = [J10['obs'][k]['first_lin_0.9'] for k, _, _ in OBS12]; f7 = [J10['obs'][k]['first_lin_0.7'] for k, _, _ in OBS12]
        d = [dd for _, dd, _ in OBS12]
        P(f'JEDI-linear (E10 numbers, these 12 observables; stages P g S h4 h5 z = 0..5, never = 6): first stage R² ≥ 0.9 {f9}, ≥ 0.7 {f7}; '
          f'Spearman(depth, first ≥ 0.9) {list(map(float, spearmanr(d, f9)))}, (≥ 0.7) {list(map(float, spearmanr(d, f7)))}\n')
    except Exception as ex:  # noqa: BLE001
        P(f'JEDI Spearman not computed: {ex}')
    for r in runs:
        Ss = X[r]['seeds']; L = list(next(iter(Ss.values()))['test5'][OBS12[0][0]]['R2'].keys())
        P(f'\n### {r}\n')
        P('| observable | depth | ' + ' | '.join(LABEL.get(l, l) for l in L) + ' | untrained ' + ' / '.join(LABEL.get(l, l) for l in L) + ' |')
        P('|---|---|' + '---|' * (len(L) + 1))
        for oid, d, lab in OBS12:
            cells = []
            for l in L:
                cells.append(' / '.join(f'{S["test5"][oid]["R2"][l]:.3f}' for S in Ss.values()) + ' [' + ', '.join(
                    f'{S["test5"][oid]["R2_ci"][l][0]:.2f}–{S["test5"][oid]["R2_ci"][l][1]:.2f}' for S in Ss.values()) + ']')
            un = ' / '.join(f'{np.mean([S["untrained"]["test5"][oid]["R2"][l] for S in Ss.values()]):.2f}' for l in L)
            P(f'| {lab} | {d} | ' + ' | '.join(cells) + f' | {un} |')
        for sd, S in Ss.items():
            P(f'- seed {sd}: Spearman(depth, first layer with R² ≥ 0.9 / ≥ 0.7) = {S["test5"]["_spearman"]}; live units per layer {S["test5"]["_live"]}; '
              f'untrained acc {S["untrained"]["acc"]:.3f}, calls per class {S["untrained"]["class_fractions"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', help='dsets-d64 | linf-2b-d32-k8 | tf-2b-d32')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--tag', default='', help="run-directory suffix, e.g. '-e100'")
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--per_class', type=int, default=4000)
    ap.add_argument('--skip_units', action='store_true', help='skip tests 2-3 (smoke runs)')
    ap.add_argument('--untrained_units', choices=('first', 'none', 'only'), default='first',
                    help="unit ablation of the untrained network: for the first seed (default), never, or only that (no other test)")
    ap.add_argument('--part', default='', help="output suffix for split jobs, e.g. '__s0' (report / plot merge x2_<run>__*.json)")
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--report', action='store_true', help='print the Markdown tables of tests 1-5 from the JSONs')
    ap.add_argument('--runs', nargs='*', help='--plot: restrict to these run names')
    ap.add_argument('--suffix', default='', help="--plot: read x2_<run><suffix>.json (e.g. '_smoke200') and write the PNGs with the suffix")
    a = ap.parse_args()
    if a.report: report(a.runs, a.suffix)
    elif a.plot: plot(a.suffix, a.runs)
    else: main_model(a)
