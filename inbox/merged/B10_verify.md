# B10 verification: if-statement programs for non-JEDI taggers (fresh sample, extra turn and transfer controls)

Verifier report for node B10 (`interp/b10_programs_small_models.py`, report `inbox/B10_report.md`). Verify module
`interp/b10_verify.py`. Results: `results/jedi/b10_verify.json` (full B10 aggregation on the new sample),
`b10_verify_extra.json` (extra analyses, new sample), `b10_verify_extra_orig.json` (the same extra analyses on the original
B10 files), `b10_verify_prep.json`, `b10_verify_torch.json`, figure `b10_verify.png`. Logs: `logs/b10-verify*.log`.

## Conclusion

On a fresh sample of 20,000 jets, the natural-jet results of B10 reproduce. Programs of about 7,000 bits agree with each small
tagger on 87–92 % of jets, a program fitted only to the truth labels agrees on 84–90 %, and other networks' programs add
nothing over the truth program. The turn-edit interpretation does not hold. The programs change their call much more often
under the mass-lowering turn than under the control rearrangement that keeps the mass fixed, as the networks do; recall is
similar on the two edits only because it counts what fraction of the network's few flips the program reproduces. The
network-by-network pattern on the shift edit (which networks' own programs beat other networks' programs) changes between the
two samples. What is uncertain: whether the programs' extra response to the turn is a response to the mass or to some other
change the turn makes that the control does not.

## Terms

- **Program**: a MARS fit (multivariate adaptive regression splines: a sum of hinge functions max(0, ±(f − t)) of jet
  features f, and products of two hinges) to each of the network's 5 output logits; the program's call is the class with
  the largest output. K16 = 16 terms per class (about 6.2–7.2 kbit).
- **DL (description length)**: bits needed to write the program (B10 definition).
- **Same call**: share of test jets where program call = network call.
- **Flip**: call on the edited jet differs from the call on the unedited jet. **Recall** = jets where the program flips to
  the same new class as the network / jets where the network flips. **Precision** = the same count / jets where the program
  flips. **Flip-κ** = Cohen kappa between the network-flip and program-flip indicators (0 = unrelated, 1 = identical).
  **Same call on edited jets** = program call on the edited jet = network call on the edited jet.
- **Truth program**: the same recipe fitted to the truth labels (log(0.95·onehot + 0.01)), never to a network.
- **Transfer**: network A's fit0 K16 program scored against network B. **Own**: B's three K16 programs (fits 0–2, disjoint
  4,000-jet fit sets).
- **Holm**: Holm–Bonferroni correction across the 9 trained networks. **McNemar**: exact binomial test on paired per-jet
  indicators.

## Code review (interp/b10_programs_small_models.py)

No bug found that changes a number.
- Splits: one permutation, five disjoint sets (fit0/1/2, dev, test); edits only on test jets. No leakage: MARS uses dev only
  for its path, and every reported number comes from test.
- Shift adds 0.06 rad to Δη of real particles only (`* real`). turn10 and ctrl10 rewrite (Δη, Δφ) of real slots only
  (`e27w.to_x` with `np.where(real, ...)`), and they share the same base rows (asserted in the verify module). Soft uses
  `e40.emit` (four-momentum exact, resorted).
- Azimuthal features: padded slots are excluded (`real` mask in the moments and in the half-plane counts); the hardest-14 set
  uses pT rank among real particles.
- Torch networks: inputs (x − shift)/scale with padded rows set to 0, mask passed, as in training. The accuracies reproduce
  the STATUS values (new sample test: Deep Sets 79.7 / 80.3, Linformer 80.4 / 80.0, transformer 75.4 / 75.6 %).
- JEDI-linear on the new sample: 80.60 % (B10 sample 81.23 %).
- Points to note (not bugs):
  - (a) Recall, the only edit metric in the transfer comparison, rises with how often a program flips, so donors that flip
    more score higher (addressed below).
  - (b) The interpretation of the similar turn10/ctrl10 recall (below).
  - (c) The figure's left panel has no truth-program reference, which is the main comparison of the reading.

## Procedure of the rerun

- **Sample.** `interp.jedi_core.sample(per_class=4000, seed=3270)`, N = 20,000 jets. Permutation rng(3270) gives fit0, fit1
  and fit2 (4,000 each), dev 2,000 and test 6,000.
- **Code.** Same code and same seeds for edits and programs. The module patches only `SEED` and the file tag. Trained
  networks only (the untrained redraws were not rerun).
- **Edit sets.** turn10 and ctrl10 on 2,794 jets (B10: 2,814); soft on 4,611.
- **Jobs.** All 1 CPU / 2 GiB:
  - b10-verify-prep: prep + torch logits, 187 s.
  - b10-verify-f1..f4: fits, 535–1,135 s.
  - b10-verify: agg + extra, 416 s.
  - b10-verify-x0: extra on the original B10 files, 15 s.
- **Statistics.** 1,000-draw jet bootstrap (rng 3279 for the extra analyses), Holm across the 9 networks.

### Extra analysis A (missing control): do the programs respond more to the turn than to the mass-preserving control?

- **Edits.** turn10 turns the 14 hardest particles in azimuth about the jet axis so that the jet mass m falls 10 %. Each
  particle keeps its pT and its distance r from the axis. ctrl10 turns the same particles by the same angles in random
  directions, then corrects them so that the pT-weighted centroid M1 and m are unchanged.
- **Measure.** On the same jets, the per-jet flip indicator under each edit, for the network, each own K16 program and the
  truth program.
- **H0.** Flip probability is the same under turn10 and ctrl10 (exact McNemar on the paired per-jet indicators).
- **Why this control.** It holds everything the turn holds (pT, r, and the amount of movement) except the mass and the
  centroid. Comparing flip rates, rather than recall, measures how much the program responds to each edit.

### Extra analysis B: shift, own vs transfer vs truth with metrics that do not reward flipping more

Precision, flip-κ and same call on edited jets, alongside recall. Also the rank of each own fit among all 11 K16 programs
(3 own + 8 donors).

## Results

### 1. Natural jets (reproduced)

| quantity | B10 (seed 3170) | rerun (seed 3270) |
|---|---|---|
| same call, K16, mean of 3 fits | 86.9–92.2 % | 86.7–91.7 % |
| K16 fit-to-fit range | ≤ 0.9 pp | ≤ 1.1 pp |
| best member ≤ 7 kbit | 85.4–92.4 % | 84.8–91.9 % (transformer s0: 84.8 / 84.9 / 85.0) |
| truth program, same call | 85.3–90.3 % | 84.1–89.5 % |
| random-feature / shuffled-logit program | 69–73 / 19–24 % | 68.5–71.6 / 21.5–27.9 % |
| own − transfer (natural) | +0.3 to +2.3 pp, Holm p 0.009 in 8/9 (PFN s0 n.s.) | +0.04 to +2.3 pp, Holm p 0.009 in 8/9 (PFN s0 n.s.) |
| transfer − truth (natural) | −0.4 to +0.2 pp, Holm p 1 | −0.1 to +0.7 pp, Holm p ≥ 0.14 |
| program − MARS on standard observables, K16 | +1.6 to +3.1, 9/9 | +1.1 to +4.4, 9/9 |
| program − GBDT on standard observables | +0.7 to +2.7, 6/9 | +0.2 to +2.6, 6/9 |
| program − GBDT on the pool | +0.7 to +4.6, 7/9 | +1.6 to +3.9, 9/9 |

### 2. Shift (Δη + 0.06 rad on every particle), K16 recall (partly reproduced)

Recall columns: own − transfer, in pp.

| network | B10 own − transfer (Holm p) | rerun own − transfer (Holm p) | rerun own fit range | rerun transfer − truth |
|---|---|---|---|---|
| JEDI-linear | +3.0 (0.009) | +2.3 (0.009) | 8.5 | +8.6 |
| Deep Sets s0 | −3.3 (0.009) | +0.7 (0.11) | 8.7 | +3.1 |
| Deep Sets s1 | +2.5 (0.009) | +1.1 (0.021) | 2.4 | +4.5 |
| Linformer s0 | −5.2 (0.009) | −6.0 (0.009) | 5.8 | −2.8 (0.009) |
| Linformer s1 | +0.1 (0.80) | −1.4 (0.012) | 16.5 | +4.3 |
| transformer s0 | +4.6 (0.009) | +5.2 (0.009) | 2.3 | +5.7 |
| transformer s1 | −5.9 (0.009) | +0.8 (0.11) | 4.0 | +9.9 |
| PFN s0 | −4.0 (0.009) | −2.2 (0.009) | 4.7 | +6.2 |
| PFN s1 | −0.9 (0.046) | +2.5 (0.009) | 2.7 | +7.8 |

- **Own − transfer beyond the fit range.** In B10 this held for JEDI-linear and transformer s0; in the rerun, for
  transformer s0 only. JEDI-linear's fit range is 8.5 pp in the rerun (fits 54.3 / 58.2 / 49.7 %).
- **Own < transfer.** 5/9 networks in B10; 3/9 in the rerun (Linformer s0, Linformer s1, PFN s0). The sign changes between
  the samples in 4/9 networks (Deep Sets s0, transformer s1, PFN s1, Linformer s1).
- **Transfer > truth.** 9/9 networks in B10; 8/9 in the rerun. For Linformer s0 the truth program reproduces more shift
  flips than the transfer programs (40.1 vs 37.3 %) and than its own programs (31.3 %).
- **Fit-to-fit range in shift recall.** Up to 16.5 pp in the rerun (Linformer s1: 40.8–57.3 %), against 9.1 pp in B10.
- **Other metrics (analysis B).** With precision, flip-κ or same call on the shifted jets, the sign of own − transfer differs
  from the recall sign in several networks. For example, JEDI-linear's recall difference is +3.0 / +2.3 pp, while its flip-κ
  difference is −2.1 / −2.4 pp (Holm p 0.009, both samples). Own fits rank anywhere from 1st to 11th of 11 programs.
- **Summary.** The statement that own programs are not reliably better than other networks' programs on the shift holds on
  both samples. The per-network list (which networks are above or below) does not.

### 3. Turn (m − 10 % at fixed pT, r) vs its mass-preserving control (the B10 interpretation is not supported)

Program flips under turn10 / ctrl10 on the same jets (counts of 2,814 jets in B10, 2,794 in the rerun):

| network | network flips turn / ctrl (B10; rerun) | own K16 programs, fits 0 / 1 / 2 (B10) | own K16 programs (rerun) |
|---|---|---|---|
| JEDI-linear | 130/81; 126/69 | 129/44, 225/71, 200/41 | 187/70, 151/103, 150/79 |
| Deep Sets s0 | 139/72; 129/82 | 190/30, 137/12, 236/51 | 172/30, 176/46, 190/46 |
| Deep Sets s1 | 108/47; 116/63 | 126/9, 85/48, 125/7 | 225/59, 156/44, 164/19 |
| Linformer s0 | 203/41; 206/41 | 274/63, 138/27, 202/8 | 328/44, 172/34, 286/7 |
| Linformer s1 | 204/40; 185/44 | 164/2, 325/27, 263/53 | 275/88, 273/70, 310/10 |
| transformer s0 | 288/108; 294/107 | 165/58, 268/62, 140/52 | 135/64, 161/84, 134/47 |
| transformer s1 | 254/68; 272/90 | 169/81, 245/67, 130/8 | 155/15, 248/25, 250/51 |
| PFN s0 | 163/81; 163/101 | 111/58, 52/47, 48/46 | 90/58, 32/36, 107/79 |
| PFN s1 | 198/86; 203/96 | 150/123, 161/29, 212/55 | 138/47, 219/80, 109/62 |
| truth program (same for all) | – | 77/37 | 104/21 |

- **Turn vs control, programs.**
  - H0 (equal flip probability) is rejected (exact McNemar p < 0.02) for 25/27 own fits in B10 and 26/27 in the rerun;
    p < 1e-5 for most.
  - The exceptions are PFN s0 fits 1 and 2 in B10 (p 0.54, 0.88), and PFN s0 fit 1 in the rerun (p 0.64). PFN s0 fit 2 in
    the rerun has p 0.004.
  - Programs flip on 1–11 % more of the jets under the turn than under the control. The networks flip on 1.7–6.7 % more.
- **What the similar recall means.** Recall on ctrl10 is computed over 40–108 network flips, and recall divides by the
  network's flips. So equal recall on the two edits does not mean that the programs respond equally to the two edits.
- **Turn flip-κ, own minus truth program.**
  - B10: +0.002 to +0.085, Holm p < 0.05 for Linformer s1 only.
  - Rerun: +0.000 to +0.114, Holm p < 0.05 for Deep Sets s0 and Linformer s0 only.
  - The networks where the own program beats the truth program differ between the samples.
- **Controls on turn flip-κ (rerun).** Random-feature program ≤ 0.034; shuffled-logit program ≤ 0.011; truth program
  0.029–0.133. Own programs: 0.045–0.186.
- **Competing hypotheses.**
  - (i) "The programs respond only to the azimuthal arrangement, not to the mass" predicts equal program flip rates under
    turn10 and ctrl10. Excluded in 25/27 fits in B10 and 26/27 in the rerun.
  - (ii) "The programs respond more to the mass-lowering turn" predicts turn > ctrl. Observed, also for the truth program.
  - What the two edits do not separate: the turn also changes the centroid M1 and higher azimuthal moments that the control
    holds or changes differently. So "responds to the mass" is not isolated either.

## Figure check (results/jedi/b10.png)

- **Left panel:** same call vs DL, mean of 3 fits with range bars. It is readable and its numbers match the table. It has no
  truth-program line, which is the reference for the reading's second paragraph; a horizontal band at 85–90 % would show it.
- **Right panel:** shift recall, own / transfer / truth. It matches the json. The rerun gives a different own-vs-transfer
  pattern for 4/9 networks, so the panel describes one sample.

## Corrected reading

The rerun (seed 3270) reproduces the following:
- Fitted the same way on 4,000 jets, MARS if-statement programs of about 7 kbit give the same call as Deep Sets, Linformer,
  transformer and PFN taggers on 85–92 % of 6,000 test jets.
- The transformers are about 3 pp lower.
- At matched bits, the programs agree more often than MARS on standard observables (9/9 networks), GBDT on standard
  observables (6/9) and GBDT on the pool (7/9; 9/9 in the rerun).

A program fitted only to the truth labels agrees with every trained network on 84–90 % of jets. A network's own programs add
0.3–2.8 pp over it. Other networks' programs add nothing on natural jets.

On the 0.06 rad shift, programs fitted to other networks reproduce more of a network's call changes than the truth program in
8–9/9 networks; Linformer s0 is the exception in the rerun. A network's own programs are not reliably better than other
networks' programs: which networks fall above or below changes between the two samples, the fit-to-fit spread in recall
reaches 16.5 pp, and the sign of the difference depends on the metric (recall vs flip-κ).

The azimuthal inputs let programs change their call under the hardest-14 turn: flip-κ is 0.04–0.24, above the
random-feature and shuffled controls, and mostly within reach of the truth program. Like the networks, the programs change
their call more often under the mass-lowering turn than under the mass-preserving control: 1.2 to 80 times as often, in
51 of 54 fits (exact McNemar p < 0.02). The B10 statement that
the programs respond about as much to the control, and therefore capture the arrangement and not the mass, is not supported.
Only the recall, meaning the share of the network's own flips that the program reproduces, is similar on the two edits.

## Files
- /Users/anrunw/Documents/aaron-interp/interp/b10_verify.py
- /Users/anrunw/Documents/aaron-interp/results/jedi/b10_verify.json, b10_verify_extra.json, b10_verify_extra_orig.json, b10_verify.png
- /Users/anrunw/Documents/aaron-interp/logs/b10-verify*.log
