"""P49 verification (loop C): rerun the key parts of interp/p49_lgatr_attention.py on a fresh sample and add two controls.

Fresh sample: JetClass test_20M <class>_105.root (not among L-GATr's training, validation or the P1 / P49 files), 1,000 random
rows per class, default_rng(5949), N 10,000; fit split: same files, 500 other random rows per class, default_rng(5950).
Model: the same final checkpoint (model_run0.pt of anrunw-ai-c-p5-lgatr2), loaded through P49's LGA class (same patched dense
attention, same formula fits re-done on the new fit split).

Parts rerun: base accuracy and plumbing; fit (hinge + MLP); held-out R2 and logit-part shares; whole-model substitution
(uniform, F1 floor, F1, F1c, F2); single-head knock-outs of b0h2, b0h5, b1h2; P49's top-16 knock-out set.
New controls:
  F1s / F1cs: F1 / F1c with the formula values permuted among the off-diagonal particle keys of each query row (diagonal kept).
     Same per-row value distribution and therefore the same total attention to particles vs the global / reference tokens; the
     pair geometry is destroyed. If F1's recovery comes from pair geometry, F1s falls to the F1 floor; if it comes from the
     spread of the logits (how sharply the row attends, and its share to the reference tokens), F1s matches F1.
  Block-limited substitution: the same edits only in blocks 0-4 (P49's 21 significant heads are all there) or only in 5-11,
     other blocks live (P49's open question).
  Block-matched set control: 10 random 16-head sets drawn from blocks 0-4 only, against P49's top-16 set.

  ./sync.sh && interp/p49_job.sh c-p49-verify '/j-jepa-vol/aaron-interp/env-lgatr/bin/python -m interp.p49_verify'
-> $AI_RESULTS/part/p49_verify.json (and p49_verify_smoke.json with --smoke)
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); A = ap.parse_args()
sys.argv = [sys.argv[0]] + (['--smoke'] if A.smoke else [])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interp import p49_lgatr_attention as P  # noqa: E402

P.VAL = Path('/j-jepa-vol/JetClass/Pythia/test_20M'); P.FIDX = 105
SEEDV, SEEDV_FIT = 5949, 5950
OUT = Path(os.environ.get('AI_RESULTS', 'results')) / 'part' / f'p49_verify{"_smoke" if A.smoke else ""}.json'
SIG21 = ['b0h0', 'b0h1', 'b0h2', 'b0h3', 'b0h5', 'b0h6', 'b0h7', 'b1h1', 'b1h2', 'b1h3', 'b1h5', 'b1h7', 'b2h1', 'b2h2', 'b2h3',
         'b2h4', 'b2h5', 'b2h6', 'b3h4', 'b3h7', 'b4h4']
TOP16 = ['b0h2', 'b0h5', 'b1h2', 'b1h1', 'b0h6', 'b2h5', 'b2h3', 'b2h6', 'b2h2', 'b2h4', 'b1h3', 'b1h5', 'b1h7', 'b0h0', 'b3h7', 'b0h3']
hid = lambda s: int(s[1:].split('h')[0]) * P.NH + int(s.split('h')[1])
GSH = None
log = P.log


def shuffled(pc):
    """permute pc (heads, P) among the off-diagonal particle keys of each query row; diagonal entries stay in place"""
    import torch
    rid, dv = P.CTX['rid'], P.CTX['dv']
    off = (~dv).nonzero(as_tuple=True)[0]
    key = rid[off].double() + torch.rand(len(off), generator=GSH, dtype=torch.float64).to(rid.device)
    perm = torch.argsort(key)                           # stable groups by row (rid integer, rand in [0, 1))
    assert torch.equal(rid[off][perm], rid[off]), 'shuffle left its row'
    idx = torch.arange(pc.shape[1], device=pc.device); idx[off] = off[perm]
    return pc[:, idx]


def edit(self, b, L, spec):
    """P49's _edit plus the shuffled families F1s / F1cs"""
    if not (spec.get('formula') and spec['formula'][0] in ('F1s', 'F1cs')):
        return ORIG_EDIT(self, b, L, spec)
    L = L.clone(); fam, fs, kind = spec['formula']; bi, ii, jj = P.CTX['pidx']; rid, nrow = P.CTX['rid'], P.CTX['nrow']
    hsl = slice(b * P.NH, (b + 1) * P.NH)
    pc = P.seg_center(self.predict(fs, kind, hsl).T.contiguous(), rid, nrow).T
    pc = shuffled(pc)
    if fam == 'F1cs':
        L[:, bi, ii + 3, jj + 3] = pc
        L[:, :, 3:, 0:3] = self.fit['const'][hsl][:, None, None, :]
    else:
        live = L[:, bi, ii + 3, jj + 3]; rm = live.T.contiguous(); rm = rm - P.seg_center(rm, rid, nrow)
        L[:, bi, ii + 3, jj + 3] = pc + rm.T
    return L


ORIG_EDIT = P.LGA._edit
P.LGA._edit = edit


def blocks(sp, bl):
    return {b: dict(sp) for b in bl}


def heads_spec(hs):
    d = {}
    for h in hs: d.setdefault(h // P.NH, {'uniform': []})['uniform'].append(h % P.NH)
    return d


def main():
    global GSH
    import torch
    from interp.part_core import kin, build
    GSH = torch.Generator().manual_seed(4951)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; ck = P.latest_ckpt(); log('checkpoint', ck, 'device', dev)
    J, rows = P.read_sample(SEEDV, P.PER); y = J['label'].astype(int); Jb = build(J, *kin(J))
    Jf0, rows_f = P.read_sample(SEEDV_FIT, P.PER_FIT, exclude=rows); Jf = build(Jf0, *kin(Jf0))
    assert not set(map(tuple, rows)) & set(map(tuple, rows_f))
    log('samples: main', len(y), 'fit', len(Jf['label']))
    out = dict(node='P49 verify', ckpt=ck, sample=f'test_20M <class>_105.root, {P.PER} random rows per class, default_rng({SEEDV})',
               fit_sample=f'same files, {P.PER_FIT} other rows per class, default_rng({SEEDV_FIT})', N=int(len(y)))
    arr = {}
    lg = P.LGA(ck, dev)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def step(nm, fn):
        try: fn()
        except Exception:
            log(f'SECTION {nm} FAILED'); traceback.print_exc(); out.setdefault('failed', []).append(nm)
        P.jdump(out, OUT)
    L0 = {}
    step('base', lambda: L0.__setitem__('t', P.sec_base(lg, Jb, y, None, out, arr, 'base')))
    step('fit', lambda: P.sec_fit(lg, Jf, out))

    def ev():
        P.sec_eval(lg, Jb, out, arr, 'eval')
        R = {k: np.array(v)[:, 0] for k, v in out['eval']['R2'].items()}; sh = np.array(out['eval']['share_pp'])
        sig = [hid(h) for h in SIG21]; ns = [i for i in range(P.NHEAD) if i not in sig]
        out['eval_groups'] = {k: dict(all=float(np.median(v)), sig21=float(np.median(v[sig])), not_sig=float(np.median(v[ns])),
                                      blocks0_4=float(np.median(v[:40])), blocks5_11=float(np.median(v[40:])),
                                      n_ge_0p7=int((v >= 0.7).sum()), n_lt_0p3=int((v < 0.3).sum())) for k, v in R.items()}
        out['eval_groups']['share_g1'] = dict(all=float(np.median(sh[1])), sig21=float(np.median(sh[1][sig])),
                                              blocks0_3=float(np.median(sh[1][:32])), blocks4_11=float(np.median(sh[1][32:])))
        for k in ('part_hinge', 'part_mlp', 'invE_mlp', 'inv_mlp', 'g1_only', 'part_null'): log('R2 groups', k, out['eval_groups'][k])
    step('eval', ev)

    def circ():
        y0 = L0['t']; c0 = y0.argmax(1) == y; ALL, E, LT = range(12), range(5), range(5, 12)
        runs = {}
        for tag, bl in (('all', ALL), ('b0_4', E), ('b5_11', LT)):
            runs[f'{tag}_uniform'] = blocks({'uniform': list(range(P.NH))}, bl)
            runs[f'{tag}_F1_floor'] = blocks({'formula': ('F1', None, None)}, bl)
            runs[f'{tag}_F1c_floor'] = blocks({'formula': ('F1c', None, None)}, bl)
            runs[f'{tag}_F1_part_hinge'] = blocks({'formula': ('F1', 'part', 'hinge')}, bl)
            runs[f'{tag}_F1s_part_hinge'] = blocks({'formula': ('F1s', 'part', 'hinge')}, bl)
            runs[f'{tag}_F1c_part_hinge'] = blocks({'formula': ('F1c', 'part', 'hinge')}, bl)
            runs[f'{tag}_F1cs_part_hinge'] = blocks({'formula': ('F1cs', 'part', 'hinge')}, bl)
            if tag != 'b5_11':
                runs[f'{tag}_F1_inv_hinge'] = blocks({'formula': ('F1', 'inv', 'hinge')}, bl)
                runs[f'{tag}_F1s_inv_hinge'] = blocks({'formula': ('F1s', 'inv', 'hinge')}, bl)
                runs[f'{tag}_F1_part_mlp'] = blocks({'formula': ('F1', 'part', 'mlp')}, bl)
                runs[f'{tag}_F1s_part_mlp'] = blocks({'formula': ('F1s', 'part', 'mlp')}, bl)
                runs[f'{tag}_F1c_part_mlp'] = blocks({'formula': ('F1c', 'part', 'mlp')}, bl)
                runs[f'{tag}_F2_part_hinge'] = blocks({'formula': ('F2', 'part', 'hinge')}, bl)
        res = {}; preds = {}
        for rn, sp in runs.items():
            L1 = lg.logits(Jb, sp); r = P.compare(y0, L1, y); res[rn] = r; preds[rn] = L1.argmax(1)
            log(f"{rn:24s} acc {100 * r['acc'][0]:.2f} [{100 * r['acc'][1]:.2f}, {100 * r['acc'][2]:.2f}] same call {r['same_call']:.3f} "
                f"W->Z {r['W_to_Z'][0]:.3f}->{r['W_to_Z'][1]:.3f}")
            out['circuit'] = dict(unedited_acc=float(c0.mean()), runs=res); P.jdump(out, OUT)
        # paired: formula vs its row-shuffled control, formula vs floor
        pair = {}
        for tag in ('all', 'b0_4', 'b5_11'):
            for a, bb in (('F1_part_hinge', 'F1s_part_hinge'), ('F1_inv_hinge', 'F1s_inv_hinge'), ('F1_part_mlp', 'F1s_part_mlp'),
                          ('F1c_part_hinge', 'F1cs_part_hinge'), ('F1_part_hinge', 'F1_floor'), ('F1s_part_hinge', 'F1_floor'),
                          ('F1c_part_hinge', 'F1c_floor'), ('F1c_part_mlp', 'F1c_floor')):
                ka, kb = f'{tag}_{a}', f'{tag}_{bb}'
                if ka in preds and kb in preds:
                    ca, cb = preds[ka] == y, preds[kb] == y
                    pair[f'{ka} - {kb}'] = dict(dacc=P.boot_mean(ca.astype(float) - cb.astype(float), seed=3), mcnemar=P.mcnemar(cb, ca),
                                                same_call=float((preds[ka] == preds[kb]).mean()))
                    log('paired', ka, '-', kb, pair[f'{ka} - {kb}'])
        out['paired'] = pair
        for tag in ('all', 'b0_4', 'b5_11'):
            fl = res[f'{tag}_F1_floor']['acc'][0]; a0 = float(c0.mean())
            for k in res:
                if k.startswith(tag + '_F1') and 'floor' not in k and not k.startswith(tag + '_F1c'):
                    res[k]['recovered_vs_F1_floor'] = float((res[k]['acc'][0] - fl) / max(a0 - fl, 1e-9))
            flc = res[f'{tag}_F1c_floor']['acc'][0]
            for k in res:
                if k.startswith(tag + '_F1c') and 'floor' not in k:
                    res[k]['recovered_vs_F1c_floor'] = float((res[k]['acc'][0] - flc) / max(a0 - flc, 1e-9))
        arr['circuit_preds'] = np.stack([preds[k] for k in runs]).astype(np.int8)
    step('circuit', circ)

    def knock():
        y0 = L0['t']; c0 = y0.argmax(1) == y; res = {}
        for h in ('b0h2', 'b0h5', 'b1h2', 'b6h0', 'b11h3'):
            L1 = lg.logits(Jb, heads_spec([hid(h)])); c1 = L1.argmax(1) == y
            res[h] = dict(dacc=P.boot_mean(c1.astype(float) - c0.astype(float)), mcnemar=P.mcnemar(c0, c1)); log('knock-out', h, res[h])
        Ls = lg.logits(Jb, heads_spec([hid(h) for h in TOP16])); accS = float((Ls.argmax(1) == y).mean())
        rng = np.random.default_rng(13); racc = []
        for _ in range(3 if A.smoke else 10):
            hs = rng.choice(40, 16, replace=False); racc.append(float((lg.logits(Jb, heads_spec(hs)).argmax(1) == y).mean()))
        res['top16'] = dict(acc=P.boot_mean((Ls.argmax(1) == y).astype(float)), random16_blocks0_4=racc,
                            rank_p=float((1 + sum(a <= accS for a in racc)) / (1 + len(racc))))
        log('top16', accS, 'block-matched random sets', racc)
        out['knockout'] = res
    step('knockout', knock)
    log('done')


if __name__ == '__main__':
    main()
