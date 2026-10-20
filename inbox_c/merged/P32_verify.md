# P32 verify: lepton route in ParT and ParticleNet (verifier report, 2026-10-20)

## Conclusion
In the ParT and ParticleNet models that receive particle-ID inputs, the flag on one particle decides whether a jet is called H→ℓνqq′. Removing the electron or muon flag from the lepton candidate moves about 99 % of these jets to other classes. In each ParT model, one attention head of the final class-token block reads that particle and has the largest single-head effect. The lepton information also reaches the output through the particle blocks, and the two routes back each other up: blocking the lepton in the class heads alone changes 11 % of calls.

The three-head "circuit" does not show that a few heads carry the decision. It calls 99 % of leptonic-top jets H→ℓνqq′, and its class-balanced accuracy is not above that of random three-head sets. The independent rerun on a fresh sample could not run because the cluster node failed, so all numbers come from one file and one seed.

## Status of the rerun
- Script `interp/p32v_verify.py`, synced to the cluster. It uses a fresh sample: files `_116`, `default_rng(9157)`, 3,000 H→ℓνqq′ jets + 400 per other class.
- New controls in the script:
  - e2n: the matched hadron's charged-hadron flag is set to neutral hadron. This is a flag change of the same size that involves no lepton flag.
  - e1x: e ↔ μ flavour swap.
  - A tight-pT-match subset.
  - Class-head attention split by the lepton's pT rank, and its attention in t→bℓν jets and in jets without a lepton.
  - The dominant class head alone as a circuit.
  - The dominant head plus 2 random particle heads, against 3 random heads that exclude the dominant head.
- Job `anrunw-ai-c-p32-verify` was submitted three times. Each time the pod was placed on node k8s-haosu-22 and rejected with UnexpectedAdmissionError ("GPU is lost"). The container never started.
- Excluding that node needs a change to `part_job.sh`'s node list. That is outside this loop's write scope and was not permitted, so the rerun did not run.
- To run it: add k8s-haosu-22.sdsc.optiputer.net to the NotIn list in `part_job.sh`, then:
  ```
  ./sync.sh && ./part_job.sh c-p32-verify 'python -m interp.p32v_verify --smoke && python -m interp.p32v_verify'
  ```
- In place of the rerun: a code review, plus a re-analysis of the stored P32 output (`results/part/p32_lepton_route.json`). The re-analysis is saved as `results/part/p32_verify_reanalysis.json`.

## Code review of interp/p32_lepton_route.py
No errors found in the following checks.
- **Layer and head indices.** Attention modules 0–7 are `blocks`; 8–9 are `cls_blocks`, labelled b1/b2. Class-block keys are offset by 1 for the class token, in both `kmask_for` and the capture.
- **Head decomposition.** The per-head output decomposition through `out_proj` reconstructs the attention output to within 4.6·10⁻⁵ (`reconstruction_max_abs`).
- **Residual patching.** It hooks the inputs and outputs of `blocks[0..7]`. The embedding is L0, and L0 patching of the lepton carries 1.00 of the e1 effect, as expected.
- **Padding.** Padded slots are masked by `key_padding_mask`. Means for the mean ablation are taken over real queries only. Adding the mean at padded query positions does not reach the class token.
- **Sampling.** Random rows from each class file (`rng.choice`); there is no first-k subsampling.
- **Build.** It uses `part_core.build` (η = jet_eta + sign(jet_eta)·Δη). The base jets are rebuilt as often as the edited jets.
- **Edits.** e1 and e2 change only the PID one-hot: ParT_kin, which has no flag inputs, gives exactly 0 under e1, e2 and e5. e3 keeps the jet mass to within 0.009 GeV.
- **ParticleNet.** The kNN override records and replays the three EdgeConv graphs in order.

Minor issues:
1. "Greedy top-k" is a ranking by single-head effect, not a greedy search.
2. The literal circuits keep all MLPs, the embeddings and the pair embedding. Under the user's circuit definition they are head circuits on top of intact MLPs, not circuits made of only those heads. The report says "MLPs kept", but the reading calls them "literal".
3. Per-set head lists for the random circuit controls were not stored. The worker already notes this.

## Re-analysis of the stored P32 numbers
Sample: the P32 sample, N = 10,004 (5,000 H→ℓνqq′, 556 per other class), seed 3215. The unedited accuracies on the P1 sample match: ParT_full 0.8593, ParT_kin 0.7447.

**1. The top-3 circuit's false positives are leptonic-top jets.** Circuit for ParT_full: b2h1, p7h8 and p8h6 are kept, and the other 77 heads are replaced by their mean output.
- t→bℓν jets called H→ℓνqq′: 551 of 556 (99.1 %). The full model calls 5 of 556.
- Other non-H→ℓνqq′ jets called H→ℓνqq′: 128 of 4,448 (2.9 %).
- Per-class recall: 0 for 7 of the 10 classes.
- Balanced (per-class mean) accuracy: 0.212. Random 3-head sets of the same composition: mean 0.145, 95th percentile 0.222, max 0.361; permutation p = 0.12.
- The reported 10-class accuracy of 0.55 mostly reflects the 50 % H→ℓνqq′ share of the sample.
- ParT_kinpid top-3: 536 of 556 t→bℓν jets called H→ℓνqq′; balanced accuracy 0.179 (random 95th percentile 0.198, p = 0.14).
- With n = 100 random sets, recall ≥ 0.973 was reached by 4 sets (p = 0.050). A random set contains b2h1 with probability 1/16, so about 6 of 100 would be expected to contain it. The top-3 recall is therefore consistent with b2h1 alone plus mean ablation; the rerun's dominant-head-only circuit would test this.
- Reading: with everything else at its mean, the kept heads give an "is there a lepton" output. They do not separate H→ℓνqq′ from t→bℓν.

**2. The selected head sets are not better than random sets at the 10-class task.**

| circuit (ParT_full) | balanced accuracy | random sets, mean | p (selected ≥ random) |
|---|---|---|---|
| 9 class heads | 0.748 | 0.820 | 0.92 |
| 37 Holm-significant heads | 0.470 | 0.703 | 1.0 |

The selected sets keep H→ℓνqq′ recall (0.989 vs random 0.972, p = 0.010) at the cost of the other classes.

**3. The control e2 is not a null edit.** e2 gives a second, matched hadron the lepton flag, which makes a two-lepton jet. In ParT_full, 626 of the 714 calls it changes go to t→bℓν. The e1 vs e2 asymmetry (98.7 % vs 14.5 % calls changed; McNemar 4,160 vs 9 jets) is therefore a comparison of "remove the lepton" against "add a second lepton". A flag change of the same size that involves no lepton flag (e2n in the verify script) is not yet measured. Even so, the e1 effect is far outside any plausible generic flag sensitivity: Δ log p of −7.23 against −0.40.

**4. Attention weight and causal effect differ for some heads.** In ParT_kinpid, head b2h7 puts 0.974 of its class-token attention on the lepton, but masking the lepton key in b2h7 alone changes the H→ℓνqq′ log-odds by 0.00. Only b2h4 (attention 0.993, knock-out −2.39) carries an effect. So "one class head attends to the lepton" holds for the causal effect, not for the attention pattern.

**5. The two routes are redundant and do not add.** Results for ParT_full, N = 2,000:

| lepton key masked in | Δ log-odds | calls changed |
|---|---|---|
| class heads only (16) | −6.77 | 11.0 % |
| particle-block heads only (64) | −5.47 | 6.0 % |
| all 80 heads | −13.74 | 94.7 % |

Masking the matched-hadron key in all 80 heads changes 1.3 % of calls. Neither route is needed on its own; either one is enough for most calls. The P32 phrase "reaches the output through many other heads" holds. The 37 Holm-significant heads mostly have small single-head effects: block-level knock-outs for p1–p8 are −0.15 to −0.74 log-odds and change ≤ 0.5 % of calls.

**6. ParT_kin: "identified by kinematics" is confounded with being the hardest particle.**
- The lepton is the highest-pT particle in 71 % of the jets.
- The control hadron is softer (median pT ratio about 3).
- So ParT_kin's b2h4 attention of 0.955 on the lepton vs 0.002 on the matched hadron does not separate "selects the lepton" from "selects the leading particle".
- The e4h edit (lepton pT ×0.5, 27.8 % of calls changed vs 4.3 % for the control) shows that ParT_kin responds to that particle's kinematics, but not how it picks the particle.
- The verify script's split by lepton pT rank, and its attention in jets without a lepton, address this.

**7. The H→ℓνqq′ call does not require a lepton flag in every jet.**
- H→ℓνqq′ jets with no lepton candidate (N = 73) are still called H→ℓνqq′ in 53 % of cases (ParT_full).
- Adding a muon flag to the leading hadron of QCD/H→4q/t→bqq′ jets moves only 9.2 % of their calls to H→ℓνqq′, so the flag is necessary in most H→ℓνqq′ jets but not sufficient.
- e1 changes calls even in the 13.9 % of jets where another lepton-flagged particle remains. The model reads the leading lepton candidate, not "any lepton flag in the jet".

**8. ParticleNet.** The kNN-graph result holds as stated: with the graph held fixed, 99.6 % of the e1 effect remains (−15.54 vs −15.60). EdgeConv 1 uses (Δη, Δφ), so its graph cannot change under e1 by construction.

## Figure
`results/part/p32_lepton_route.png`.
- Both panels match the JSON.
- Left panel: the ParT_kin bars for flag edits are 0 and invisible, which the title states.
- Right panel: single-head knock-outs for ParT_full, with the matched-hadron control as dots.
- The titles use declarative wording ("changes ~all"). The right panel does not show the joint knock-outs, which carry the redundancy result.

## H0/H1 summary (stored data)

| test | H0 | H1 | result |
|---|---|---|---|
| e1 vs e2 calls changed (ParT_full) | e1 and e2 change calls equally often | e1 changes more | McNemar 4,160 vs 9 jets, p < 10⁻³⁰⁰; H0 rejected |
| top-3 circuit, balanced accuracy | selected = random 3-head sets | selected > random | p = 0.12; H0 not rejected |
| class-head circuit (9), balanced accuracy | selected = random | selected > random | p = 0.92; H0 not rejected |
| 37-head circuit, balanced accuracy | selected = random | selected > random | p = 1.0; H0 not rejected |

Competing hypotheses:
- **"Only the class head reads the lepton."** Excluded: class-head masking changes 11 % of calls.
- **"Particle-block mixing alone carries it."** Excluded: particle-block masking changes 6 % of calls.
- **Redundant routes.** Consistent with all three joint knock-outs.

## Corrected reading
- **Flag edit.** In the four PID models, removing the electron/muon flag from the leading lepton candidate (e1) changes 98–99 % of H→ℓνqq′ calls, mostly to H→4q, Z, W and QCD.
- **Matched-hadron control.** Giving a matched-pT hadron a lepton flag (e2, which makes a two-lepton jet) changes 12–25 %, mostly to t→bℓν.
- **What the model reads.** It reads that specific particle: e1 acts even when another lepton-flagged particle remains. The flag is necessary but not sufficient: adding it to a hadron in QCD/H→4q/t→bqq′ jets moves 9 % of calls.
- **Dominant head.** In each ParT model, one class head (b2h1 in ParT_full, b2h4 in ParT_kinpid and ParT_kin) attends to the lepton with weight 0.96–0.99 and gives the largest single-head knock-out. Attention alone does not identify it: ParT_kinpid's b2h7 attends with weight 0.97 and has no effect.
- **Redundant routes.** The lepton reaches the class token through two redundant routes: the class head's direct read and the particle blocks. Masking either route alone changes 6–11 % of calls; masking both changes 95 %.
- **Head circuits.** With MLPs kept and the other heads mean-ablated, the head sets keep H→ℓνqq′ recall. They call almost all t→bℓν jets H→ℓνqq′ and do not exceed random head sets in balanced accuracy, so no small head set that keeps the 10-class behaviour has been found.
- **ParT_kin.** It responds to the lepton's kinematics (pT halving changes 28 % of calls). Whether its class head selects the lepton or simply the highest-pT particle is not yet separated.
- **ParticleNet.** The flag effect does not pass through the kNN graph.
- **Scope.** One file, one seed; the fresh-sample rerun is pending.

## Explainer entry (for site/explain.py)
- **Question.** How do ParT and ParticleNet recognise H→WW*→ℓνqq′ jets?
- **Background.** These jets usually contain an electron or muon. The models with particle-ID inputs receive five one-hot flags per particle.
- **Method.** Edit one input and compare the model's call before and after, over the same jets.
  - Remove the lepton flag; as a control, flag a hadron of similar pT.
  - Block the class token's attention to the lepton in chosen heads by setting the logit $a_{h,\ell} \to -10^9$.
  - Keep only some heads and replace the others by their mean output, $\bar o_h = \mathbb{E}_{\text{jets, tokens}}[o_h]$.
  - The effect is measured as $\Delta \log\frac{p_{\ell\nu qq}}{1-p_{\ell\nu qq}}$ per jet and as the share of calls changed, with bootstrap intervals and McNemar tests.
- **Result.**
  - Removing the flag changes about 99 % of calls.
  - One class head carries the largest single effect.
  - Blocking the lepton in the class heads alone, or in the particle blocks alone, changes ≤ 11 % of calls; blocking both changes 95 %.
  - A three-head circuit calls 99 % of leptonic-top jets H→ℓνqq′.
- **Checks.** Matched-hadron controls; random head sets (n = 100); a model without flag inputs gives exactly zero response to flag edits.
- **Why it matters.** It is the first non-mass mechanism found in ParT. The lepton route is redundant rather than one small circuit, so small-circuit descriptions of ParT need class-balanced tests.
