# P22 — What in JetClass training produces ParT's axis response (claims C3s, C10)

Status: **running** (training). Jobs `anrunw-ai-c-p22-t`, `-t2`, `-u`, `-a2` (logs `/j-jepa-vol/aaron-interp/logs/c-p22-<arm>.log`), launched 2026-10-20 10:56–11:13 UTC. They should finish about 13:00–13:40 UTC (10 epochs at 10–13 min per epoch). The analysis script is written and smoke-tested. Run it next round with:

```
./sync.sh && ./part_job.sh c-p22-run 'python -m interp.p22_planted_true_part'
```

This writes `results/part/p22_planted_true_run.json` and `.png`.

## Conclusion

Conclusion: four Particle Transformers are training on 1.5 million W, Z and QCD jets with the setup of the earlier planted-label runs. Two learn the true W/Z/QCD labels (two seeds). One learns the true labels while every training jet is shifted by a random small offset (up to 0.08 rad) from its stored axis. One repeats the mass-window label with a second seed. The scripts run end to end, and all four jobs are training with the GPU 84–100 % busy. After one epoch, the true-label network classifies 69.5 % of held-out W/Z/QCD jets correctly. There is no result on the axis response yet. Whether training on true labels at this scale reproduces the released models' response (R about 0.75) is open until the analysis runs on the finished checkpoints.

## Question and predictions (fixed before any trained checkpoint is analysed)

R is the change in the network's read-out under a rigid φ + 0.06 rad shift of every constituent about the stored jet axis, divided by the change under a matched real mass change. The invariant mass is unchanged by the shift; the axis moment m_A = √(M0·M2) rises by 12 %. R is the per-jet least-squares slope, with a 95 % bootstrap interval over jets. Known values: released ParT_full 0.79 and ParT_kin 0.72 (P14 control, P16v); planted m_inv-window arm a 0.185 (P16); untrained initialisations −0.23 to 0.78 (P11v, 8 inits).

| hypothesis | R_t (true labels) | R_u (true labels + offsets) | accuracy u vs t |
|---|---|---|---|
| H_thesis: the true JetClass labels produce the response | 0.6–0.8 | < 0.3 | within 1 pt |
| H_alt1: data scale or 10 classes are needed | ≈ R_a ≈ 0.2 | – | – |
| H_alt2: augmentation leaves the response (an offset-invariant feature correlated with the axis moment) | – | ≈ R_t | – |

Tests:
- H0: R_t = R_a; H0: R_u = R_t; H0: R_t = R_t2 (seed spread); H0: R_a = R_a2; H0: R_t = R_released_kin; H0: R_u = R_init0. Each is a permutation test that swaps the two models' per-jet (Δd(φ), Δd(dilation)) pairs (2,000 permutations), with Holm correction over the six.
- McNemar on 3-class correctness, t vs u.
- McNemar on the W → Z call indicator (true-W jets) before vs after φ, Holm over all models and edits.

## Procedure

**Training** (`interp/p22_planted_true_train.py`, imports the P14 code path `interp/p14_planted_train.py`):
- Data: 1.5M jets = 5 random train_100M files per class of WToQQ / ZToQQ / ZJetsToNuNu (default_rng(14), the same files as P14). The same 30,000 validation rows (default_rng(140)) and the same 20,000 held-out test rows of test_20M *_100 (default_rng(14)).
- Architecture: weaver-core 0.4.17 ParticleTransformer with the authors' configuration and ParT_kin inputs (pt_log, e_log, logptrel, logerel, deltaR, deta, dphi; pair features from the four-vectors).
- Optimiser: Ranger, lr 1e-3, batch 512, AMP, 10 epochs, learning rate decayed over epochs 7–9. All as P14.

| arm | label | outputs | seed (init / batch order) | augmentation |
|---|---|---|---|---|
| t | true class (QCD, Z, W) | 3 | 14 / default_rng(1000+epoch), = P14 arm a | none |
| t2 | true class | 3 | 15 / default_rng(15000+epoch) | none |
| u | true class | 3 | 14 / as t | per jet and batch: offset c = \|c\|(cos ψ, sin ψ), \|c\| ~ U(0, 0.08) rad, ψ ~ U(0, 2π) (default_rng((14, epoch, batch))), added to every real particle's (Δη, Δφ) about the stored axis; px, py, pz, E rebuilt with η = η_jet + sign(η_jet)·Δη, φ = φ_jet + Δφ (part_core.build convention); jet p_T and energy recomputed. Validation and test jets not offset |
| a2 | 1[\|m_inv − 85\| < 10 GeV] (P14 side file, row-matched by assertion) | 2 | 15 / as t2 | none |

Checkpoints: `/j-jepa-vol/aaron-interp/models/p22/<arm>/epoch-<e>.pt`, `final.pt`, `log.json` (per-epoch held-out accuracy and confusion).

**Analysis** (`interp/p22_planted_true_part.py`, imports the P14 / P16 battery from `interp/p14_planted_part.py`):
- Sample: fresh rows of test_20M/{ZJetsToNuNu, ZToQQ, WToQQ}_104.root, 3,000 random rows per class (default_rng(2204 + class index)), N = 9,000.
- Models: t, t2, u, a2, P14 arm a, released ParT_kin and ParT_full, and four untrained 3-output ParT_kin (torch seeds 100–103) as the null range.
- Edits: φ + 0.06; matched dilation (per-jet scale k_j giving the same Δlog m_A as φ); angles × 1.1; whole-jet rotation including the axis (null).
- Metrics: accuracy and 3×3 confusion (released: 10-class and W/Z/QCD-restricted, plus the P1-sample check 0.859 / 0.745); R with bootstrap CI and the rotation null; R on W jets; ridge-probe R; W → Z before / after φ with bootstrap CI and McNemar; route shares of the dilation and × 1.1 edits (P2 method: per-particle inputs only vs pair four-vectors only); DAS at blocks 2 and 8 (P8 method, 4 random directions as control). For the 3-output and released models, DAS uses W/Z jets, call = argmax(ℓ_Z, ℓ_W), and target W-like = 1[m_inv < 85.8 GeV] or 1[m_A < c_A] at a matched rate.

Compute: 4 CPU, 16 Gi, 1 GPU per job via `interp/p10_part_seeds_job.sh`, the same wrapper P14 used. It is used instead of `part_job.sh` because training allocates 21 GiB of GPU memory, so the GPU list is restricted to cards with ≥ 24 GB.

## Numbers so far

Smoke tests:
- Training smoke (`c-p22-smoke2`: arms u and a2, 1 file per class, 60 steps): both code paths run.
- Analysis smoke (`c-p22-asmoke`: smoke checkpoints, 300 jets per class) runs end to end, including JSON output, permutation tests and DAS. Its numbers come from untrained or barely trained checkpoints and are not results. On those 900 fresh jets, released ParT_kin gives R 0.727 [0.684, 0.771] and W → Z 15.7 % → 38.7 %, consistent with P16v (0.719) and C3 (16 → 37 %).

Edit sizes on the smoke sample:

| edit | Δlog m_inv | Δlog m_A |
|---|---|---|
| φ + 0.06 | 0.0000 | +0.118 |
| matched dilation | +0.114 | +0.118 |
| rotation null | 0 | 0 |

Augmentation check (arm u, 2,000 pool jets, full run):
- Max relative change of m_inv: 9.0e-4. The η part of the offset is a shift in pseudorapidity, which is not an exact boost for massive constituents.
- Mean Δlog m_A: +0.118.

Training at the time of writing:

| arm | GPU | steps/s | GPU util | epoch 0 held-out accuracy | epoch 0 confusion (rows true, cols call) |
|---|---|---|---|---|---|
| t | V100-32GB | 4.9–5.2 | 80–88 % | 0.695 (3-class) | QCD [6046, 301, 319], Z [1341, 3993, 1325], W [836, 1977, 3862] |
| a2 | A10 | 3.8–3.9 | 84–100 % | 0.832 (own label; P14 arm a epoch 0: 0.857) | out [9520, 2240], in [1128, 7112] |
| t2 | A10 | 3.6–3.7 | 100 % | epoch 0 running | – |
| u | A10 | 3.8 | 84–100 % | epoch 0 running | – |

## Known limitation for the analysis

On W/Z jets, the two DAS targets (m_inv below or above the W/Z midpoint vs m_A at the matched rate) disagree on about 1 % of jets (smoke sample: 0.010). This is the same near-collinearity P16v found for labels a and b (97.7 % agreement). DAS will therefore have few discriminating pairs and little power to separate an m_inv from an m_A variable. R and the route split are the primary read-outs.

## Explainer entry (draft; result and figure to be added after the analysis)

- **Question.** The released ParT changes its W-vs-Z decision when all constituents are rotated 0.06 rad about the stored jet axis, although the jet's invariant mass is unchanged. A ParT trained by us on an invariant-mass window label does so much less. Is the response produced by training on the real W/Z/QCD labels, and does training with random axis offsets remove it?
- **Background.** ParT's per-particle inputs Δη, Δφ and ΔR are measured from the stored jet axis. An angular moment $m_A = \sqrt{M_0 M_2}$, with $M_0 = \sum_i p_{T,i}$ and $M_2 = \sum_i p_{T,i}\,\Delta R_i^2$, rises under a rigid offset. The invariant mass $m^2 = (\sum_i p_i)^2$ does not.
- **Method.** $R = \sum_j \Delta d_j^{\phi}\,\Delta d_j^{\rm dil} / \sum_j (\Delta d_j^{\rm dil})^2$, where $d = \ell_W - \ell_Z$, the edit φ is $\Delta\phi_i \to \Delta\phi_i + 0.06$, and the edit dil is $(\Delta\eta_i, \Delta\phi_i) \to k_j(\Delta\eta_i, \Delta\phi_i)$ with $k_j^2 = M_0M_2(\phi)/M_0M_2$. $R = 0$: the network reads only $m$; $R = 1$: it reads $m_A$.
- **Result / checks / why it matters.** To be filled from `p22_planted_true_run.json`. Checks: seed pair (t, t2), untrained range, rotation null. Why it matters: it separates the training label from the architecture as the source of the axis convention in C3.

## Files

- `/Users/anrunw/Documents/aaron-interp/interp/p22_planted_true_train.py`
- `/Users/anrunw/Documents/aaron-interp/interp/p22_planted_true_part.py`
- cluster: `/j-jepa-vol/aaron-interp/models/p22/{t,t2,u,a2}/`, smoke output `/j-jepa-vol/aaron-interp/results/part/p22_planted_true_smoke.json`
