# P34 verify — lepton circuit follow-up (skeptic rerun on a fresh file, plus three missing controls)

## Conclusion

On a fresh set of jets, removing the electron/muon label from one particle again changes about 98 % of ParT's H→ℓνqq′ calls, while the same kind of label change on a hadron of similar pT changes under 1 %. Adding a lepton label to a hadron in a jet with no lepton turns 18 % of those jets into H→ℓνqq′ calls (0.2 % for the control). The new test does not support "the class-token MLPs compute the call": with all 80 attention heads intact, replacing those two MLPs by their average output leaves the H→ℓνqq′ call almost unchanged (recall 0.975 vs 0.980). They are needed only after almost every head has been replaced by its average. Within that reduced network, only one head (the last class-attention block's lepton-reading head, b2h1) produces the call. Other heads whose scores separate the class equally well produce none. Which units turn the lepton signal into the call in the intact network is still open.

## What was checked

**Code review.** I read `interp/p34_lepton_circuit.py`, `interp/p32v_verify.py` (unchanged since commit af4d385), `interp/p34_job.sh` and the P32 helpers it imports. No bug was found in the parts that produce the reported numbers:

- **Head indices.** `lab2ih('b2h1')` = (9, 0) = `cls_blocks[1]`, head 0. `DOM` matches. The class-block key index is offset by +1 for the class token (in `kmask_for` and `vzero`).
- **Per-head decomposition.** Each head's output after the output projection is reconstructed to a maximum absolute error of 9e-5 (`head_recon_max_abs`).
- **Mean ablation.** The mean is taken over real query tokens only (`realq` comes from `key_padding_mask`), and padded tokens are never used as keys. The MLP hook replaces the `fc2` output, before dropout and the residual connection.
- **Sampling.** Rows are random (`load_rows` with `rng.choice`), never the first k.
- **η sign.** `part_core.build` uses η = jet_η + sign(jet_η)·Δη.
- **Edits.** e1 changes only the flag columns: px, py, pz, E, charge, d0 and dz are bit-identical (checked in P34 and here). The lepton's slot index in `rebuild(BL)` equals the slot found in `BL` for 100 % of jets (new check).
- **Job script.** `p34_job.sh` differs from `part_job.sh` only by a node-preference block, and `part_job.sh` already excludes k8s-haosu-09.

**Minor point.** Circuits whose output should be constant ("nothing kept", particle-head-only circuits) give AUC 0.46–0.49 instead of exactly 0.5. The cause is floating-point residuals of about 1e-4 in the mean replacement, not signal. Because these circuits are part of the k = 1 random distribution, that null sits slightly below 0.5.

**Figure check** (`results/part/p34_lepton_circuit.png`, read). Both panels match the JSON. The right panel leaves out the all-80-heads variants. One of them, "all 80 heads, class-block MLPs at mean" (recall 0.973, t→bℓν called H→ℓνqq′ 0.008 in P34's own JSON), runs against the "class MLPs compute the call" reading and was also left out of the P34 report text.

## Rerun: procedure and sample

- **Job.** `anrunw-ai-c-p34-verify` (node suncave-10, 503 s, exit 0). Script `interp/p34v_verify.py` (new). It reuses P34's `Run`, `metr` and `knockouts` without changing them.
- **Sample.** JetClass test_20M `<sample>_117.root` (P32 used _115; P32v and P34 used _116), `numpy.default_rng(7341)`, random rows. 1,000 jets per class for evaluation (N = 10,000) and a disjoint 300 per class as the mean-ablation reference set.
- **Lepton jets.** 988 H→ℓνqq′ jets have a lepton candidate (the highest-pT particle with isElectron or isMuon) and a matched charged hadron. 50 % of the candidates are electrons, and the candidate is the leading particle in 69 % of jets.
- **Unedited 10-class accuracy on this sample.** ParT_full 0.863, ParT_kinpid 0.796, PN_full 0.845. On the P1 sample (from P34): ParT_full 0.859, ParT_kin 0.745.
- **Statistics.** 1,000 bootstrap resamples over jets give the 95 % intervals in brackets. McNemar is the exact paired test on per-jet call changes. "Empirical p" = (1 + number of random sets ≥ selected) / (1 + number of sets).

## A. Flag edits on the input jet

What each edit does to the jet:

- **e1:** the lepton candidate's isElectron/isMuon is set to 0 and isChargedHadron to 1. Its four-momentum, charge and track displacements are unchanged.
- **e2n (control):** a charged hadron with matched pT gets isChargedHadron = 0 and isNeutralHadron = 1. This is a one-hot change of the same size on a particle of similar pT, with no lepton flag involved.
- **e1x:** electron ↔ muon on the lepton.
- **Tight subset (new):** jets whose matched hadron's pT is within a factor 1.5 of the lepton's (n = 295). Over all jets, the median pT ratio between hadron and lepton is e^1.0 ≈ 2.7, so the general control is loosely matched.

**H0:** e1 and e2n change the same share of calls. **H1:** e1 changes more.

| model | e1 calls changed | e2n | e2n, tight subset | e1x | McNemar e1 vs e2n (jets changed only by e1 / only by e2n) |
|---|---|---|---|---|---|
| ParT_full | 97.8 % [96.8, 98.7] | 0.71 % [0.20, 1.21] | 0.34 % [0.0, 1.0] | 0.20 % [0.0, 0.51] | 959 / 0, p < 1e-280 |
| ParT_kinpid | 97.7 % [96.7, 98.6] | 2.33 % [1.52, 3.34] | 3.39 % [1.69, 5.43] | 0.30 % | 943 / 1, p < 1e-280 |
| PN_full | 98.3 % [97.4, 99.1] | 1.32 % [0.71, 2.02] | 1.69 % [0.34, 3.39] | 0.20 % | 960 / 2, p < 1e-280 |

Δlog p(H→ℓνqq′) under e1 (change in the log of the H→ℓνqq′ probability): ParT_full −7.31 [−7.45, −7.18] (P34: −7.23). H0 is rejected. Competing hypotheses and their predictions:

- **"Any flag change on a pT-matched particle":** e1 ≈ e2n. Excluded, including on the tight subset.
- **"Flavour identity":** e1x should change calls as often as e1 does. Excluded (0.2 % vs 97.8 %).

**Sufficiency (new).** Sample: jets of QCD, H→4q, Z→qq, W→qq and t→bqq′ with no lepton candidate (n = 3,819). The leading charged hadron's flag is set to muon (s_mu) or electron (s_e). The control (s_n) sets the same particle to neutral hadron. Share of jets called H→ℓνqq′:

| model | before | s_mu | s_e | s_n (control) | McNemar s_mu vs s_n (jets called H→ℓνqq′ only after s_mu / only after s_n) |
|---|---|---|---|---|---|
| ParT_full | 0.16 % | 18.3 % [17.1, 19.6] | 17.1 % | 0.16 % | 693 / 2, p < 1e-200 |
| ParT_kinpid | 0.18 % | 4.7 % [4.0, 5.4] | 4.3 % | 0.21 % | 175 / 4 |
| PN_full | 0.21 % | 13.9 % [12.9, 15.1] | 14.3 % | 0.18 % | 526 / 1 |

Δlog-odds(H→ℓνqq′) under s_mu (change in log p/(1−p)): ParT_full +5.32, PN_full +5.26, ParT_kinpid +3.69, against +0.13 to +0.19 under s_n. In ParT_full, s_mu changes 41 % of calls, but only 18 % go to H→ℓνqq′ and 0.7 % to t→bℓν; where the rest go was not recorded. Per class (ParT_full): Z→qq 37 %, W→qq 25 %, QCD 15 %, H→4q 9 %, t→bqq′ 0.5 %.

Reading: the flag is necessary for about 98 % of H→ℓνqq′ calls. Adding it raises the H→ℓνqq′ score by about 5 in log-odds but makes the call in only a minority of jets from other classes.

## B. Knock-outs inside ParT_full (replicated)

Edit: the attention logit of the lepton's key is set to −1e9 in the named heads, so those heads cannot read the lepton. The control masks the matched hadron's key instead.

| knock-out | calls changed (lepton key masked) | Δlog-odds | matched-hadron control |
|---|---|---|---|
| b2h1 alone | 3.4 % [2.3, 4.6] | −5.47 | 0.0 % |
| 16 class heads | 13.3 % [11.1, 15.4] | −6.86 | 0.0 % |
| 64 particle heads | 6.9 % [5.4, 8.5] | −5.46 | 0.4 % |
| all 80 heads | 95.7 % [94.4, 97.0] | −13.77 | 0.6 % |

Increment from adding b2h1 to k random heads (10 sets each): k = 1: +3.6 % [3.4, 3.8]; k = 7: +5.0 %; k = 31: +27.8 % [20.2, 36.9]. All 30 sets are Holm-significant (Holm correction over the sets at each k; McNemar per set). This matches P34: +3.0 % to +22.4 %, 12.6 / 5.8 / 95.1 %.

## C. Literal circuit (ParT_full and ParT_kinpid)

**Literal circuit:** the network with only the listed attention heads computing. Every other head's output, and the output of all 10 MLPs (8 particle blocks + 2 class-token blocks), is replaced by its average over the reference jets. The input embedding, the LayerNorms and the final layer are kept.

**Replicated:**

| ParT_full | b2h1 alone | top-3 (b2h1, p7h8, p8h6) |
|---|---|---|
| H→ℓνqq′ recall, literal | 0.000 (every jet called Z→qq) | 0.000 |
| AUC of H→ℓνqq′ log-odds, vs the other 9 classes | 0.922 | 0.923 |
| AUC vs t→bℓν | 0.426 | 0.431 |
| recall with the 2 class-block MLPs kept | 0.812 [0.787, 0.836] | 0.847 [0.822, 0.869] |
| t→bℓν called H→ℓνqq′, class-block MLPs kept | 0.894 [0.875, 0.913] | 0.931 |

ParT_kinpid, b2h4: AUC 0.912; recall with class-block MLPs kept 0.618, and with those MLPs 0.799 of t→bℓν jets are called H→ℓνqq′ (P34: 0.914, 0.643, 0.791).

**C3 (new): are the class-block MLPs needed in the intact network?** All 80 heads kept; MLP variants.

| ParT_full | recall H→ℓνqq′ | t→bℓν called H→ℓνqq′ | balanced acc. |
|---|---|---|---|
| all MLPs kept (full network) | 0.980 [0.971, 0.988] | 0.010 | 0.863 |
| 2 class-block MLPs at mean | **0.975 [0.964, 0.984]** | 0.006 | 0.827 |
| 8 particle-block MLPs at mean | 0.934 | 0.136 | 0.744 |
| all 10 MLPs at mean | 0.859 | 0.115 | 0.603 |

ParT_kinpid: 0.943 → 0.926 with the class-block MLPs at mean.

- **H0 (class-block MLPs needed for the call):** predicts recall falls toward 0. **Observed:** a 0.5-point drop, with overlapping intervals.
- P34's own JSON shows the same (0.983 → 0.973).
- The class-block MLPs are therefore needed only once almost every head has been replaced by its mean. In the intact network, the other heads and the particle-block MLPs make the call without them.
- Rejecting t→bℓν depends on the particle-block MLPs (0.010 → 0.136 when they are at mean).

**C2 (new): is the literal circuit's "no call" missing information or an offset?** The circuit's H→ℓνqq′ log-odds is thresholded once. The threshold is set on the reference jets so that 10 % of them (the class prior) are called H→ℓνqq′. Results on the evaluation sample (ParT_full, b2h1 alone):

- recall 0.368 [0.339, 0.400];
- t→bℓν called H→ℓνqq′ 0.496;
- other jets with a lepton candidate called 0.006;
- other jets without one called 0.000.

Top-3: recall 0.392; kinpid b2h4: 0.382. The literal circuit's score does make H→ℓνqq′ calls under a fixed read-out, so recall 0 under argmax reflects the mean-ablated logits never reaching the top class. With this threshold, the circuit calls more t→bℓν jets than H→ℓνqq′ jets.

**C1 (new): is the call specific to the lepton heads?** Single class heads alone, with the 2 class-block MLPs kept (16 runs), and 60 random composition-matched 3-head sets (1 class head + 2 particle heads) with the class-block MLPs kept. **H0:** a random set reaches the same recall.

- **Single class heads.** ParT_full: only b2h1 gives recall > 0.01 (0.81). b2h6 and b2h8 give literal AUC 0.91 and recall 0.00. ParT_kinpid: only b2h4 (0.62); b2h7, with AUC 0.90, gives 0.00.
- **Random 3-head sets, ParT_full.** The 7 sets that contain b2h1 give 0.80–0.83, and the 53 without it give at most 0.009. Empirical p for top-3 = 0.016 (ParT_kinpid 0.049).
- **Literal-circuit AUC, top-3 vs the same 60 random sets.** p = 0.082 (ParT_full) and 0.033 (ParT_kinpid), against P34's 0.010 with 100 sets. The AUC advantage of the selected 3 heads over random sets does not replicate at α = 0.05 for ParT_full.

Reading: the AUC is shared by several class heads (b2h1, b2h6, b2h8). Producing the call through the class-block MLPs is specific to one head (b2h1; b2h4 in kinpid). The two added particle heads change recall by +0.035 (0.812 → 0.847).

## Issues with the P34 reading

1. **"The heads carry the lepton signal and the class-block MLPs compute the call"** (conclusion, thesis note). This is not supported for the network. With all heads present, the class-block MLPs can be replaced by their mean at almost no cost (0.980 → 0.975; P34's own JSON: 0.983 → 0.973). It holds only inside the circuit where 77–79 of 80 heads are mean-ablated.
2. **"It never makes the call."** True under argmax only. One fixed threshold on the same circuit's score gives recall 0.37 with no jets without a lepton called.
3. **Top-k AUC vs random sets.** P34's k = 3 result (p 0.010) gives p 0.082 on a fresh file (ParT_full). The specificity that does replicate belongs to the single head b2h1, and it shows in the call with the class-block MLPs kept, not in the AUC.
4. **e2n control.** It is loosely pT-matched (median ratio about 2.7). The tight subset gives the same answer, so the e1 result stands.
5. **Missing figure panel.** The figure's right panel lacks the "all 80 heads, class MLPs at mean" bar, which is the comparison that bears on the MLP claim.

## Corrected reading

- **Flag edits.** Flag presence on the leading lepton is necessary for 98 % of H→ℓνqq′ calls in the four models with particle ID. Flavour does not matter (e↔μ 0.2 %). A pT-matched charged→neutral change does not reproduce the effect (0.3–3.4 %, tight match). Adding a flag to a hadron in no-lepton jets raises the H→ℓνqq′ log-odds by about 5 and the share of H→ℓνqq′ calls from 0.2 % to 18 % (ParT_full).
- **Routes.** The flag reaches the class token through redundant head routes: class heads 13 %, particle heads 7 %, both 96 %; b2h1 adds +3.6 % to +28 % on top of random heads.
- **Literal circuit.**
  - The lepton heads alone give a lepton-presence score (AUC 0.92 vs the rest, 0.43 vs t→bℓν). Argmax never reaches H→ℓνqq′, but a fixed threshold gives recall 0.37.
  - In this reduced network, the two class-block MLPs turn b2h1's (not b2h6's or b2h8's) output into the call (recall 0.81), and the call takes in t→bℓν jets too.
  - In the intact network the class-block MLPs are not needed for the call (0.980 → 0.975). The particle-block MLPs are needed to reject t→bℓν.

## Explainer entry (for site/explain.py; not written there, outside this loop's write scope)

**Question.** Does ParT's H→ℓνqq′ decision come from a small set of attention heads that read the lepton, and which part of the network turns that reading into the decision?

**Background.**

- H→WW*→ℓνqq′ jets contain an isolated electron or muon, and ParT receives per-particle flags (isElectron, isMuon, …).
- An attention head mixes information between particles. The class token is an extra vector that collects information in the last two "class-attention" blocks, and the output layer reads it.
- Each block also has an MLP: a small two-layer network applied to every token.

**Method.**

- **(1) Input edits.** Set the lepton's flag to charged hadron, with every other input unchanged. Compare with the same-size change on a hadron of similar pT.
- **(2) Literal circuit.** Keep a few heads computing and replace every other head and MLP output by its average over reference jets. This "mean ablation" removes their jet-by-jet information:
  $$h_{\text{other}}(x) \to \bar h = \tfrac{1}{N_{\text{ref}}}\sum_{j} h(x_j).$$
- **(3) Necessity test.** Replace only the class-block MLPs by their mean, with all heads kept.

Metrics:

- Recall = share of true H→ℓνqq′ jets called H→ℓνqq′.
- AUC = probability that a random H→ℓνqq′ jet has a higher H→ℓνqq′ log-odds $\ell = \log\frac{p}{1-p}$ than a random jet of another class.
- McNemar's test compares paired per-jet outcomes.

**Result** (figure: `results/part/p34_lepton_circuit.png`).

- Edit (1) changes 97.8 % of calls (control 0.7 %).
- With one head (b2h1) kept, the score separates the class (AUC 0.92), but no jet is called H→ℓνqq′ unless the class-block MLPs are kept (recall 0.81).
- With all heads kept, removing those MLPs leaves recall at 0.975.

**Checks.**

- Fresh file and seed.
- A tight pT-matched control.
- Electron ↔ muon swap.
- A sufficiency edit: adding a flag to a hadron in no-lepton jets.
- 16 single class heads and 60 random 3-head sets.
- A threshold read-out of the reduced circuit.

**Why it matters.** The lepton flag is the input that the H→ℓνqq′ class depends on. The step that turns it into a decision is not located in one place: in a heavily ablated network the class-token MLPs do it, while in the full network other heads and MLPs can do it without them.

**Glossary additions:**

- **Mean ablation:** replacing a unit's output by its average over reference jets, which removes its jet-specific information.
- **Literal circuit:** a network in which only the named units compute and all others are mean-ablated.
- **Threshold read-out:** a decision made by one fixed cut on a score, with the cut set on separate reference jets.
- **Class-block MLP:** the feed-forward layer applied to the class token in the two class-attention blocks.
- **Sufficiency edit:** adding the feature to jets that lack it, to see whether it alone moves the decision.

## Files

- `interp/p34v_verify.py`
- `results/part/p34_verify.json` and `results/part/p34_verify_smoke.json`
- log: `/j-jepa-vol/aaron-interp/logs/c-p34-verify.log`
