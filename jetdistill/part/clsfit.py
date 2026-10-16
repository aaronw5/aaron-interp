"""The class fit: ParT's per-particle embeddings (the input of its two class-attention blocks) written as formulas of
each particle's own inputs, fed through ParT's own class-attention blocks, final LayerNorm and last layer (all fixed).

  python -m jetdistill.part.clsfit extract [n_fit n_dev]   # embeddings of the first n_fit training / n_dev validation jets

extract: ParT is run again from the ROOT files (float32 inputs) for the rows of the cached 'fit' and 'dev' jets (the same
order); results/_cls/<net>/<which>/: x_f16.npy (J, 128, 128) the embeddings (0 in empty slots), mask.npy, z.npy, L.npy."""
import json, sys, time
import numpy as np
from .. import config
from ..config import DATA, RESULTS


def log(*a):
    print(*a, flush=True)


def rows_of(which, n):
    S = np.load(DATA / 'splits.npz'); return S[which][:n]


def extract(net='full', n_fit=20000, n_dev=10000, log=log):
    from .data import read_root
    from .network import ParTNetwork
    files = json.loads((DATA / 'train' / 'files.json').read_text()); off = np.cumsum([0] + [f['jets'] for f in files])
    model = ParTNetwork(net); Lc = np.load(DATA / 'train' / f'net_{net}.npz')['L']
    for which, n in ((w, k) for w, k in (('fit', n_fit), ('dev', n_dev)) if k):
        t0 = time.time(); r = rows_of(which, n); d = RESULTS / '_cls' / str(net) / which; d.mkdir(parents=True, exist_ok=True)
        X = np.lib.format.open_memmap(d / 'x_f16.npy', 'w+', np.float16, (len(r), 128, 128)); M = np.zeros((len(r), 128), bool)
        Z = np.zeros((len(r), 128), np.float32); L = np.zeros((len(r), 10), np.float32)
        for fi, f in enumerate(files):                     # every file once: its rows among r
            pos = np.flatnonzero((r >= off[fi]) & (r < off[fi + 1]))
            if not len(pos): continue
            J = read_root(f['file']); J = {k: v[r[pos] - off[fi]] for k, v in J.items()}
            R = model.run_internals(J); X[pos] = R['x']; M[pos] = R['mask']; Z[pos] = R['z']; L[pos] = R['logits']
            log(f'  {which}: {f["file"]}: {len(pos)} jets, {time.time() - t0:.0f} s')
        X.flush(); np.save(d / 'mask.npy', M); np.save(d / 'z.npy', Z); np.save(d / 'L.npy', L)
        agree = (L.argmax(1) == Lc[r].argmax(1)).mean(); dl = np.abs(L - Lc[r]).max()
        log(f'{which}: {len(r)} jets in {time.time() - t0:.0f} s; logits vs the stored ones: max |diff| {dl:.1e}, same class {100 * agree:.2f}%')


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'extract': extract('full', *(int(x) for x in a))      # n_dev = 0: skip; the validation split is stored sorted: take all of it
    elif cmd == 'tune': tune('full', *(int(x) for x in a))           # [dim n_fit steps]


# ---------------------------------------------------------------- the per-particle formula and its tuning
CTX = ['mass', 'sum_pt', 'n_particles', 'tau21', 'tau32', 'tau43', 'C2', 'D2', 'sd_mass', 'sj2_dr', 'n_sd0_above_3', 'n_lepton', 'jet_charge',
       'z_1st', 'girth', 'mass_displaced3', 'lep_z', 'dc_n', 'sv_n', 'ak02_n']        # jet quantities every particle sees (context)
PFEAT = ['ln pT', 'ln E', 'ln pT/pT_jet', 'ln E/E_jet', 'ΔR', 'Δη', 'Δφ', 'charge', 'charged hadron', 'neutral hadron', 'photon', 'electron', 'muon',
         'tanh d0', 'σ(d0)', 'tanh dz', 'σ(dz)', 'ln(1 + pT rank)']


def particle_features(J, rows, ctx=CTX):
    """(J, 128, F) float32: each particle's 18 inputs (ParT's 17 and its pT rank) and the jet quantities; the mask"""
    x = np.asarray(J['x'][rows], np.float32); e = np.asarray(J['ext'][rows], np.float32); jet = np.asarray(J['jet'][rows], np.float32)
    pt, deta, dphi = x[..., 0], x[..., 1], x[..., 2]; E, q, typ, d0, d0e, dz, dze = (e[..., i] for i in range(7)); ok = pt > 0
    lg = lambda a: np.log(np.maximum(a, 1e-8))
    F = [lg(pt), lg(E), lg(pt / jet[:, :1]), lg(E / jet[:, 3:4]), np.hypot(deta, dphi), deta, dphi, q] + [(typ == t).astype(np.float32) for t in (1, 2, 3, 4, 5)] + \
        [np.tanh(d0), np.clip(d0e, 0, 1), np.tanh(dz), np.clip(dze, 0, 1), np.log1p(np.arange(128, dtype=np.float32))[None].repeat(len(rows), 0)]
    C = [np.asarray(J['Q'][k], np.float32)[rows][:, None].repeat(128, 1) for k in ctx]
    return (np.stack(F + C, -1) * ok[..., None]).astype(np.float32), ok


def knots_of(F, ok, n=9):
    """the candidate thresholds of each feature: its 5-95 % quantiles over the real particles (fewer for discrete features)"""
    P = F[ok]; return [np.unique(np.quantile(P[:, f], np.linspace(.05, .95, n))).astype(np.float32) for f in range(F.shape[-1])]


def basis(F, knots):
    """(…, K) per particle: 1, each feature, max(0, f − t) and max(0, t − f) at its thresholds (torch or numpy)"""
    import torch
    cols = [torch.ones_like(F[..., :1])]
    for f, kn in enumerate(knots):
        v = F[..., f:f + 1]; T = torch.as_tensor(kn, device=F.device)
        cols += [v, torch.clamp(v - T, min=0), torch.clamp(T - v, min=0)]
    return torch.cat(cols, -1)


def cls_forward(model, xr, mask):
    """ParT's own class-attention blocks, final LayerNorm and last layer on the embeddings xr (N, P, 128): (neurons, logits)"""
    x = xr.permute(1, 0, 2); pm = ~mask; cls = model.cls_token.expand(1, x.size(1), -1)
    for b in model.cls_blocks: cls = b(x, x_cls=cls, padding_mask=pm)
    h = model.norm(cls).squeeze(0); return h, model.fc(h)


def tune(net='full', dim=32, n_fit=100000, steps=800, lr=1e-4, lam=0.01, lam_e=1.0, chunk=8000, device='mps', log=log, out=None):
    """the class fit: least squares of the top-`dim` directions of the normalized embeddings on the per-particle terms
    (20k extracted training jets), then all coefficients tuned through ParT's frozen class blocks toward its probabilities
    (+ λ·R on the neurons), full batch (chunks of jets of similar multiplicity), the best validation step every 50"""
    import torch
    from ..pipeline import jets
    from .network import ParTNetwork
    t0 = time.time(); D = RESULTS / '_cls' / str(net); model = ParTNetwork(net).model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    nz = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    Jf, Jd = jets(net, 'fit'), jets(net, 'dev')
    # directions of the normalized embeddings, the least-squares start (on the extracted training jets)
    Xe = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r'); Me = np.load(D / 'fit' / 'mask.npy'); ne = min(len(Me), 20000)   # the start: 20k jets (~1M particles)
    Xe, Me = Xe[:ne], Me[:ne]
    Fe, oke = particle_features(Jf, np.arange(ne)); assert (oke == Me).all()
    Y = nz(np.asarray(Xe, np.float32))[oke]; mu = Y.mean(0); V = np.linalg.svd(Y[::5] - mu, full_matrices=False)[2][:dim]
    knots = knots_of(Fe, oke); Fr = Fe[oke]; Sr = (Y - mu) @ V.T; G = 0; b = 0
    for i in range(0, len(Fr), 100000):     # the normal equations in float64 on the CPU (on the GPU, float32 sums over ~1M particles make the solution blow up)
        Bc = basis(torch.from_numpy(Fr[i:i + 100000]), knots).numpy().astype(np.float64); G = G + Bc.T @ Bc; b = b + Bc.T @ Sr[i:i + 100000]
    dsc = np.sqrt(np.maximum(np.diag(G), 1e-12)); Bt = St = None
    W0 = np.linalg.lstsq(G / dsc[:, None] / dsc[None] + 1e-8 * np.eye(len(dsc)), b / dsc[:, None], rcond=None)[0] / dsc[:, None]
    K = G.shape[0]; Np = len(Fr); del Xe, Y, Fr, Sr
    # whitening: tuned in a basis where the terms are uncorrelated with unit spread (an invertible change of coefficients);
    # the hinge terms of one feature are strongly correlated, and Adam steps each coefficient on its own
    ev, Qe = np.linalg.eigh(G / dsc[:, None] / dsc[None]); ev = np.maximum(ev, 1e-6 * ev.max())     # near-duplicate terms: a floor
    Mw = (Qe / np.sqrt(ev)[None] / dsc[:, None]) * np.sqrt(Np)        # basis @ Mw: whitened terms; basis @ W = (basis @ Mw) @ U
    U0 = (np.sqrt(ev)[:, None] * (Qe.T @ (dsc[:, None] * W0))) / np.sqrt(Np)
    log(f'class fit: {len(PFEAT)} particle inputs + {len(CTX)} jet quantities, {K} terms per particle, {dim} directions; start {time.time() - t0:.0f} s')
    # the jets: features (GPU, float16), mask, targets
    def prep(J, n):
        r = np.arange(min(n, len(J['y']))); F, ok = particle_features(J, r); o = np.argsort(ok.sum(1), kind='stable')     # similar multiplicity together
        return dict(F=torch.from_numpy(F[o]).to(device, torch.float16), ok=torch.from_numpy(ok[o]).to(device), o=o, n=len(r),
                    P=torch.from_numpy(np.asarray(J['P'][r][o], np.float32)).to(device), H=torch.from_numpy(np.asarray(J['H'][r][o], np.float32)).to(device),
                    net=J['net'][r][o], y=J['y'][r][o])
    tr, va = prep(Jf, n_fit), prep(Jd, len(Jd['y']))
    Xe = np.load(D / 'fit' / 'x_f16.npy', mmap_mode='r'); assert len(Xe) >= tr['n'], 'extract the embeddings of the tuning jets first'
    St = torch.empty((tr['n'], 128, dim), dtype=torch.float16, device=device); oi = tr['o']
    for a in range(0, tr['n'], 5000):
        rr = oi[a:a + 5000]; srt = np.argsort(rr); x = nz(np.asarray(Xe[rr[srt]], np.float32))[np.argsort(srt)]
        St[a:a + 5000] = torch.from_numpy(((x - mu) @ V.T).astype(np.float16)).to(device)
    sv = torch.from_numpy(((nz(np.asarray(Xe[:5000], np.float32))[np.load(D / 'fit' / 'mask.npy')[:5000]] - mu) @ V.T).var(0).astype(np.float32) + 1e-6).to(device)
    nreal = float(tr['ok'].sum()); del Xe
    Vt = torch.from_numpy(V.astype(np.float32)).to(device); mut = torch.from_numpy(mu.astype(np.float32)).to(device)
    vn = tr['H'].var(0) + 1e-6
    Mt = torch.from_numpy(Mw.astype(np.float32)).to(device)
    W = torch.from_numpy(U0.astype(np.float32)).to(device).requires_grad_(True)      # coefficients of the whitened terms
    def run(S, a, b_, Wq):
        ok = S['ok'][a:b_]; P = int(ok.sum(1).max()); F = S['F'][a:b_, :P].float(); m = ok[:, :P]
        Sh = basis(F, knots) @ (Mt @ Wq); xr = (Sh @ Vt + mut) * m[..., None]     # (n, P, 128) embeddings (0 in empty slots)
        return (*cls_forward(model, xr, m), Sh, m, P)
    def score():
        with torch.no_grad():
            pred = np.concatenate([run(va, a, a + chunk, W)[1].argmax(1).cpu().numpy() for a in range(0, va['n'], chunk)])
        return float((pred == va['net']).mean()), float((pred == va['y']).mean())
    m_, v_ = torch.zeros_like(W), torch.zeros_like(W); s0 = score(); best = (s0[0], W.detach().clone(), 0); path = [(0, *s0)]
    log(f'  least squares: validation same class as ParT {100 * s0[0]:.2f}%, accuracy {100 * s0[1]:.2f}%, {time.time() - t0:.0f} s')
    out = out or RESULTS / 'cls_fit' / str(net); out.mkdir(parents=True, exist_ok=True)
    def save(best, path):
        np.savez(out / 'cls_fit.npz', W=(Mt @ best[1]).cpu().numpy(), V=V, mu=mu)        # the coefficients of the plain terms
        (out / 'cls_fit.json').write_text(json.dumps(dict(dim=dim, terms=K, n_fit=tr['n'], lam=lam, lam_e=lam_e, lr=lr, steps=steps, best_step=best[2], path=path, ctx=CTX, pfeat=PFEAT,
                                                         knots=[k.tolist() for k in knots])))
    for i in range(steps):
        g = torch.zeros_like(W)
        for a in range(0, tr['n'], chunk):
            Wq = W.detach().requires_grad_(True); h, L, Sh, m, P = run(tr, a, a + chunk, Wq)
            loss = -(tr['P'][a:a + chunk] * torch.log_softmax(L, 1)).sum() / tr['n'] + lam * ((h - tr['H'][a:a + chunk]) ** 2 / vn).sum() / (tr['n'] * 128)
            if lam_e > 0: loss = loss + lam_e * (((Sh - St[a:a + chunk, :P].float()) ** 2 / sv) * m[..., None]).sum() / (nreal * dim)   # closeness to ParT's own embeddings
            g += torch.autograd.grad(loss, Wq)[0]
        with torch.no_grad():
            m_.mul_(.9).add_(.1 * g); v_.mul_(.999).add_(.001 * g * g)
            W.sub_(lr * (m_ / (1 - .9 ** (i + 1))) / (torch.sqrt(v_ / (1 - .999 ** (i + 1))) + 1e-8))
        if device == 'mps': torch.mps.empty_cache()
        if i % 50 == 49 or i == steps - 1:
            s = score(); path.append((i + 1, *s))
            if s[0] > best[0]: best = (s[0], W.detach().clone(), i + 1); save(best, path)     # the best so far on disk (a crash loses nothing)
            log(f'  step {i + 1}: validation same class as ParT {100 * s[0]:.2f}%, accuracy {100 * s[1]:.2f}%, {time.time() - t0:.0f} s')
    save(best, path)
    log(f'class fit: best validation same class as ParT {100 * best[0]:.2f}% (step {best[2]}), {time.time() - t0:.0f} s')
    return best
