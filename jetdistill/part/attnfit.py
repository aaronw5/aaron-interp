"""The layer after class attention, written with an attention of its own: ParT's 128 neurons as combinations of particle
terms pooled by 8 heads, each head a softmax weighting over the jet's particles (as ParT's class attention weights its
particles). Then ParT's own last layer.

  token i      the particle's 18 inputs (ParT's 17 and its pT rank) and 20 jet quantities, as hinge terms φᵢ (clsfit.basis)
  head h       score sₕᵢ = gₕ(particle i) + φᵢ·aₕ  (gₕ a fixed physics start: hardest, uniform, near the axis, charged,
               displaced, leptons, photons, most energetic; aₕ learned); as in ParT's class attention the head also attends
               to the class token itself, with a learned constant score bₕ: αₕ = softmax over [bₕ, sₕ₁, …, sₕₙ], so the
               weight α_cls,h = 1 − Σᵢ αₕᵢ is jet-dependent (how much the particles count at all)
  neurons      Σₕ [Σᵢ αₕᵢ φᵢ, α_cls,h]·Wₕ + c (α_cls,h·Wₕ,last: the class token's own value), then ParT's last layer
  2nd layer    as ParT's second class block: the class token is now jet-dependent (block 1's output), the particles the same.
               z₁ = layer 1's summary (R numbers, its pooled terms·W₁); scores sₕᵢ = φᵢ·Aₕ·[z₁, 1], the class token's own
               score bₕ·[z₁, 1]; the neurons take both layers' pooled terms and self-weights
Fit: least squares of ParT's neurons on the pooled terms (the start weightings), then everything tuned toward ParT's
probabilities + λ·R on its neurons, full batch, terms whitened (Adam on correlated hinge terms is unstable).

  python -m jetdistill.part.attnfit tune [n_fit steps]"""
import json, sys, time
import numpy as np
from ..config import RESULTS
from .clsfit import particle_features, knots_of, basis, CTX, PFEAT, log

HEADS = ['hardest (ln pT)', 'uniform', 'near the axis (−ΔR)', 'charged', 'displaced (|tanh d0|)', 'leptons', 'photons', 'most energetic (ln E)']


def start_scores(F):
    """(…, 8) the fixed start of each head's score from the particle inputs (the PFEAT order of clsfit.particle_features)"""
    import torch
    lnpt, lne, dr, q = F[..., 0], F[..., 1], F[..., 4], F[..., 7]; lep = F[..., 11] + F[..., 12]; pho = F[..., 10]; td0 = F[..., 13]
    return torch.stack([lnpt, torch.zeros_like(lnpt), -5 * dr, 2 * (q != 0).float(), 3 * td0.abs(), 3 * lep, 2 * pho, lne], -1)


def tune(net='full', n_fit=100000, steps=800, lr=1e-3, lam=0.01, chunk=4000, n_ls=40000, layers=2, R=16, device='mps', log=log, out=None):
    import torch
    from ..pipeline import jets
    from .network import ParTNetwork
    t0 = time.time(); H8 = len(HEADS); Jf, Jd = jets(net, 'fit'), jets(net, 'dev')
    Kw = torch.from_numpy(np.asarray(ParTNetwork(net).last[0], np.float32)).to(device); bw = torch.from_numpy(np.asarray(ParTNetwork(net).last[1], np.float32)).to(device)
    # thresholds and whitening of the per-particle terms (20k jets)
    F0, ok0 = particle_features(Jf, np.arange(20000)); knots = knots_of(F0, ok0)
    Fr = F0[ok0]; Np = len(Fr); G = 0                                      # in float64 on the CPU (float32 GPU sums over ~1M particles are not enough)
    for i in range(0, Np, 100000): Bc = basis(torch.from_numpy(Fr[i:i + 100000]), knots).numpy().astype(np.float64); G = G + Bc.T @ Bc
    K = G.shape[0]; del Fr
    d = np.sqrt(np.maximum(np.diag(G), 1e-12)); ev, Qe = np.linalg.eigh(G / d[:, None] / d[None]); ev = np.maximum(ev, 1e-6 * ev.max())
    Mt = torch.from_numpy(((Qe / np.sqrt(ev)[None] / d[:, None]) * np.sqrt(Np)).astype(np.float32)).to(device)     # whitened terms: basis @ Mt
    def prep(J, n):
        r = np.arange(min(n, len(J['y']))); F, ok = particle_features(J, r); o = np.argsort(ok.sum(1), kind='stable')
        return dict(F=torch.from_numpy(F[o]).to(device, torch.float16), ok=torch.from_numpy(ok[o]).to(device), n=len(r), o=o,
                    P=torch.from_numpy(np.asarray(J['P'][r][o], np.float32)).to(device), H=torch.from_numpy(np.asarray(J['H'][r][o], np.float32)).to(device),
                    net=J['net'][r][o], y=J['y'][r][o])
    tr, va = prep(Jf, n_fit), prep(Jd, len(Jd['y'])); vn = tr['H'].var(0) + 1e-6
    A = torch.zeros((K, H8), device=device)                                  # learned part of the head scores (whitened terms)
    bc = torch.zeros(H8, device=device)                                      # each head's score for the class token itself
    A2 = torch.zeros((K, H8, R + 1), device=device); b2 = torch.zeros((H8, R + 1), device=device)    # layer 2: scores bilinear in the particle terms and z₁
    def pooled(S, a, b_, A_, bc_):
        ok = S['ok'][a:b_]; P = int(ok.sum(1).max()); F = S['F'][a:b_, :P].float(); m = ok[:, :P]
        Phi = basis(F, knots) @ Mt                                           # (n, P, K) whitened terms
        sc = (start_scores(F) + Phi @ A_).masked_fill(~m[..., None], -1e9)  # (n, P, 8)
        sc = torch.cat([bc_.expand(len(m), 1, -1), sc], 1); al = torch.softmax(sc, 1)    # (n, 1 + P, 8): the class token itself, then the particles
        pp = torch.einsum('nph,npk->nhk', al[:, 1:], Phi)                    # (n, 8, K) the particles' pooled terms
        return torch.cat([pp, al[:, 0, :, None]], 2).reshape(len(m), -1)    # (n, 8 (K + 1)): + each head's self-weight
    def pooled2(S, a, b_, z, A2_, b2_):                                     # layer 2: the class token's query and own key depend on z₁
        ok = S['ok'][a:b_]; P = int(ok.sum(1).max()); F = S['F'][a:b_, :P].float(); m = ok[:, :P]
        Phi = basis(F, knots) @ Mt; z1 = torch.cat([z, torch.ones_like(z[:, :1])], 1)            # (n, R + 1)
        Q2 = torch.einsum('khr,nr->nkh', A2_, z1)                            # (n, K, 8) each jet's score direction
        sc = (start_scores(F) + torch.einsum('npk,nkh->nph', Phi, Q2)).masked_fill(~m[..., None], -1e9)
        sc = torch.cat([(z1 @ b2_.T)[:, None], sc], 1); al = torch.softmax(sc, 1)
        pp = torch.einsum('nph,npk->nhk', al[:, 1:], Phi)
        return torch.cat([pp, al[:, 0, :, None]], 2).reshape(len(m), -1)
    def lsq(X, Y):
        Gx, bx = 0, 0                                                         # float64 on the CPU
        for i in range(0, len(X), 5000):
            Xc = torch.cat([X[i:i + 5000], torch.ones_like(X[i:i + 5000, :1])], 1).cpu().numpy().astype(np.float64); Gx = Gx + Xc.T @ Xc; bx = bx + Xc.T @ Y[i:i + 5000].cpu().numpy().astype(np.float64)
        dx = np.sqrt(np.maximum(np.diag(Gx), 1e-12)); return np.linalg.lstsq(Gx / dx[:, None] / dx[None] + 1e-6 * np.eye(len(dx)), bx / dx[:, None], rcond=None)[0] / dx[:, None]
    nl = min(n_ls, tr['n']); rng = [(a, min(a + chunk, nl)) for a in range(0, nl, chunk)]; Yl = tr['H'][:nl]
    # least-squares start: layer 1 (start weightings) → ParT's neurons; z₁ = the top R directions of that prediction
    with torch.no_grad():
        X1 = torch.cat([pooled(tr, a, b_, A, bc) for a, b_ in rng]); Wl = lsq(X1, Yl)
        Hp = (X1 @ torch.from_numpy(Wl[:-1].astype(np.float32)).to(device) + torch.from_numpy(Wl[-1].astype(np.float32)).to(device)).cpu().numpy().astype(np.float64)
        hm = Hp.mean(0); _, _, Vt = np.linalg.svd(Hp - hm, full_matrices=False); hs = (Hp - hm) @ Vt[:R].T; zs = hs.std(0) + 1e-6
        W1m = Wl[:-1] @ Vt[:R].T / zs; c1 = (Wl[-1] - hm) @ Vt[:R].T / zs           # z₁ = pooled1 @ W1 + c1 (standardized)
        W1 = torch.from_numpy(W1m.astype(np.float32)).to(device); c1 = torch.from_numpy(c1.astype(np.float32)).to(device)
        if layers == 2:
            X2 = torch.cat([pooled2(tr, a, b_, X1[a:b_] @ W1 + c1, A2, b2) for a, b_ in rng]); X1 = torch.cat([X1, X2], 1); del X2
        W0 = lsq(X1, Yl); del X1
    W = torch.from_numpy(W0[:-1].astype(np.float32)).to(device); c = torch.from_numpy(W0[-1].astype(np.float32)).to(device)
    log(f'attention fit: {len(PFEAT)} particle inputs + {len(CTX)} jet quantities, {K} terms per particle, {layers} layer(s) of {H8} heads (+ class-token self-weights), {W.numel() + A.numel() + bc.numel() + (layers == 2) * (A2.numel() + b2.numel() + W1.numel() + R)} coefficients; start {time.time() - t0:.0f} s')
    params = [A, bc, W, c] + ([A2, b2, W1, c1] if layers == 2 else []); [p.requires_grad_(True) for p in params]
    def run(S, a, b_, ps):
        X = pooled(S, a, b_, ps[0], ps[1])
        if layers == 2: X = torch.cat([X, pooled2(S, a, b_, X @ ps[6] + ps[7], ps[4], ps[5])], 1)
        h = X @ ps[2] + ps[3]; return h, h @ Kw + bw
    def score(ps):
        with torch.no_grad():
            pred = np.concatenate([run(va, a, a + chunk, ps)[1].argmax(1).cpu().numpy() for a in range(0, va['n'], chunk)])
        return float((pred == va['net']).mean()), float((pred == va['y']).mean())
    s0 = score(params); best = (s0[0], [p.detach().clone() for p in params], 0); path = [(0, *s0)]
    log(f'  least squares: validation same class as ParT {100 * s0[0]:.2f}%, accuracy {100 * s0[1]:.2f}%, {time.time() - t0:.0f} s')
    out = out or RESULTS / 'attn_fit' / str(net); out.mkdir(parents=True, exist_ok=True)
    names = ['A', 'b_cls', 'W', 'c', 'A2', 'b2_cls', 'W1', 'c1']
    def save(best, path):
        np.savez(out / 'attn_fit.npz', M=Mt.cpu().numpy(), **{k: v.cpu().numpy() for k, v in zip(names, best[1])})    # A, A2 in whitened terms (plain: M @ A)
        (out / 'attn_fit.json').write_text(json.dumps(dict(heads=HEADS, layers=layers, R=R, terms=K, n_fit=tr['n'], lam=lam, lr=lr, steps=steps, best_step=best[2], path=path, ctx=CTX, pfeat=PFEAT,
                                                           knots=[k.tolist() for k in knots])))
    m_ = [torch.zeros_like(p) for p in params]; v_ = [torch.zeros_like(p) for p in params]
    for i in range(steps):
        gs = [torch.zeros_like(p) for p in params]
        for a in range(0, tr['n'], chunk):
            ps = [p.detach().requires_grad_(True) for p in params]; h, L = run(tr, a, a + chunk, ps)
            loss = -(tr['P'][a:a + chunk] * torch.log_softmax(L, 1)).sum() / tr['n'] + lam * ((h - tr['H'][a:a + chunk]) ** 2 / vn).sum() / (tr['n'] * 128)
            for g, gg in zip(gs, torch.autograd.grad(loss, ps)): g += gg
        with torch.no_grad():
            for p, g, m, v in zip(params, gs, m_, v_):
                m.mul_(.9).add_(.1 * g); v.mul_(.999).add_(.001 * g * g); p.sub_(lr * (m / (1 - .9 ** (i + 1))) / (torch.sqrt(v / (1 - .999 ** (i + 1))) + 1e-8))
        if device == 'mps': torch.mps.empty_cache()
        if i % 50 == 49 or i == steps - 1:
            s = score(params); path.append((i + 1, *s))
            if s[0] > best[0]: best = (s[0], [p.detach().clone() for p in params], i + 1); save(best, path)     # the best so far on disk
            log(f'  step {i + 1}: validation same class as ParT {100 * s[0]:.2f}%, accuracy {100 * s[1]:.2f}%, {time.time() - t0:.0f} s')
    save(best, path)
    log(f'attention fit: best validation same class as ParT {100 * best[0]:.2f}% (step {best[2]}), {time.time() - t0:.0f} s')
    return best


if __name__ == '__main__':
    cmd, *a = sys.argv[1:]
    if cmd == 'tune': tune('full', *(int(x) for x in a))
