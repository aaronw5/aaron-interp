"""Small taggers for the hls4ml jet dataset, written to be read as well as trained.

All models take
    x    : (B, N, 3) float, normalised (pT, deta, dphi), padded rows set to 0
    mask : (B, N) bool, True where a real particle sits
and return logits (B, 5).

Three families (names as used on the command line and in the output directories):
    tf-{B}b-d{D}          native pre-LN transformer encoder, B blocks, width D, 2 heads, MLP x2, GELU,
                          no positional encoding, masked mean pooling, linear head
    linf-{B}b-d{D}-k{K}   same, but keys and values are projected from N tokens to K with one learned (N x K)
                          matrix E, shared across heads and between K and V (Linformer, key-value sharing);
                          padded tokens are zeroed before the projection
    dsets-d{D}            Deep Sets: per-particle MLP 3 -> D -> D (ReLU), masked sum, MLP D -> D -> 32 -> 16 (ReLU),
                          linear 5. Float JEDI-like, no context round.
    jedis-d{D}            (X8, opt-in) float JEDI-style tagger with a context round: per-particle MLP 3 -> D -> D (ReLU) = p_i,
                          g = masked mean of p_i over the real particles, second per-particle MLP [p_i, g] (2D) -> D -> D (ReLU)
                          = u_i, masked sum S = sum_i u_i, MLP D -> 32 -> 16 (ReLU), linear 5.

Attention is written out (separate q, k, v Linear layers, explicit softmax) so that per-head weights and attention
maps can be read off directly. Compatible with torch >= 1.13.
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9  # score for masked keys (finite, so a fully masked row cannot produce NaN)


class Attention(nn.Module):
    """Multi-head self-attention with a key padding mask; optional Linformer projection of K and V."""

    def __init__(self, d, heads, n_tokens=None, k=None):
        super().__init__()
        assert d % heads == 0
        self.d, self.heads, self.dh = d, heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.linformer_k = k
        if k is not None:
            assert n_tokens is not None
            # E: (N, k), shared across heads and between K and V
            self.E = nn.Parameter(torch.randn(n_tokens, k) / math.sqrt(n_tokens))

    def forward(self, x, mask, return_attn=False):
        B, N, _ = x.shape
        H, dh = self.heads, self.dh
        q = self.q(x).view(B, N, H, dh).transpose(1, 2)  # (B, H, N, dh)
        k = self.k(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v(x).view(B, N, H, dh).transpose(1, 2)
        if self.linformer_k is not None:
            m = mask[:, None, :, None].to(k.dtype)
            k = torch.einsum("bhnd,nk->bhkd", k * m, self.E)  # (B, H, k, dh)
            v = torch.einsum("bhnd,nk->bhkd", v * m, self.E)
            scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, H, N, k); no key mask after projection
        else:
            scores = q @ k.transpose(-1, -2) / math.sqrt(dh)  # (B, H, N, N)
            scores = scores.masked_fill(~mask[:, None, None, :], NEG)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.d)
        out = self.proj(out)
        return (out, attn) if return_attn else out


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, d, heads, mlp_mult, n_tokens=None, k=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, n_tokens=n_tokens, k=k)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(), nn.Linear(mlp_mult * d, d))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerTagger(nn.Module):
    def __init__(self, n_particles, d=16, blocks=1, heads=2, mlp_mult=2, n_in=3, n_classes=5, cls_token=False,
                 linformer_k=None, final_ln=True):
        super().__init__()
        self.n_particles, self.cls_token = n_particles, cls_token
        n_tokens = n_particles + (1 if cls_token else 0)
        self.embed = nn.Linear(n_in, d)
        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_mult, n_tokens=n_tokens, k=linformer_k) for _ in range(blocks)])
        self.final_ln = nn.LayerNorm(d) if final_ln else nn.Identity()
        self.head = nn.Linear(d, n_classes)

    def tokens(self, x, mask):
        """Embedded tokens and their mask (with the CLS token prepended when used)."""
        h = self.embed(x)
        if self.cls_token:
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
        return h, mask

    def pool(self, h, mask):
        if self.cls_token:
            return h[:, 0]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, x, mask):
        h, mask = self.tokens(x, mask)
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.final_ln(h)
        return self.head(self.pool(h, mask))


class DeepSetsTagger(nn.Module):
    def __init__(self, d=64, rho=(32, 16), n_in=3, n_classes=5):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        widths = [d, d] + list(rho)
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.rho = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x, mask):
        p = self.phi(x) * mask.unsqueeze(-1).to(x.dtype)
        return self.head(self.rho(p.sum(1)))


class JediStyleTagger(nn.Module):
    """X8: float JEDI-style tagger. phi1 per particle (3 -> d -> d), context g = masked mean of phi1 over real particles,
    phi2 per particle on [phi1_i, g] (2d -> d -> d), masked sum, rho d -> 32 -> 16, head 16 -> 5 (all ReLU except the head)."""

    def __init__(self, d=64, rho=(32, 16), n_in=3, n_classes=5):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.phi2 = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        widths = [d] + list(rho)
        layers = []
        for a, b in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.rho = nn.Sequential(*layers)
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x, mask):
        mm = mask.unsqueeze(-1).to(x.dtype)
        p = self.phi(x) * mm
        g = p.sum(1) / mm.sum(1).clamp(min=1.0)
        u = self.phi2(torch.cat([p, g.unsqueeze(1).expand_as(p)], -1)) * mm
        return self.head(self.rho(u.sum(1)))


_TF = re.compile(r"^tf-(\d+)b-d(\d+)$")
_LINF = re.compile(r"^linf-(\d+)b-d(\d+)-k(\d+)$")
_DSETS = re.compile(r"^dsets-d(\d+)$")
_JEDIS = re.compile(r"^jedis-d(\d+)$")


def parse_model(name):
    """'tf-1b-d16' -> dict of constructor args (without n_particles / cls_token)."""
    m = _TF.match(name)
    if m:
        return dict(arch="tf", blocks=int(m.group(1)), d=int(m.group(2)), heads=2, mlp_mult=2)
    m = _LINF.match(name)
    if m:
        return dict(arch="linf", blocks=int(m.group(1)), d=int(m.group(2)), heads=2, mlp_mult=2, k=int(m.group(3)))
    m = _DSETS.match(name)
    if m:
        return dict(arch="dsets", d=int(m.group(1)), rho=[32, 16])
    m = _JEDIS.match(name)
    if m:
        return dict(arch="jedis", d=int(m.group(1)), rho=[32, 16])
    raise ValueError(f"unknown model name {name!r}; expected tf-{{B}}b-d{{D}}, linf-{{B}}b-d{{D}}-k{{K}}, dsets-d{{D}} or jedis-d{{D}}")


def build_model(spec, n_particles, cls_token=False, final_ln=True):
    """spec: dict from parse_model (plus optional overrides). Returns an nn.Module."""
    arch = spec["arch"]
    if arch == "tf":
        return TransformerTagger(n_particles, d=spec["d"], blocks=spec["blocks"], heads=spec["heads"],
                                 mlp_mult=spec["mlp_mult"], cls_token=cls_token, final_ln=final_ln)
    if arch == "linf":
        return TransformerTagger(n_particles, d=spec["d"], blocks=spec["blocks"], heads=spec["heads"],
                                 mlp_mult=spec["mlp_mult"], cls_token=cls_token, linformer_k=spec["k"],
                                 final_ln=final_ln)
    if arch == "dsets":
        return DeepSetsTagger(d=spec["d"], rho=tuple(spec["rho"]))
    if arch == "jedis":
        return JediStyleTagger(d=spec["d"], rho=tuple(spec["rho"]))
    raise ValueError(arch)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
