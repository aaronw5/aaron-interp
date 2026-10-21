# E26 verification (verifier report, 2026-10-20)

## Conclusion
On a new set of 10,000 jets, the E26 numbers repeat to within about 2 percentage points. When the jet is boosted to 10 % lower pT
at fixed mass, holding the network's summed-pT object (S0) at its pre-boost value still removes about 57 % of the changed calls,
and holding two learned directions that carry ΣpT ($M_0$) and the pT-weighted angular spread ($M_2$) removes 85 %. The random-direction
control in E26 was weaker than reported: 69 % of each random direction fell on inactive coordinates that the next layer ignores.
Restricting the random directions to the active coordinates raises the null maximum from 0.44 to 0.48 (1 dim) and from 0.40 to
0.56 (4 dims). The named holds still exceed this null at $\lambda = 0.9$. Two things are still uncertain. First, generic
high-variance directions also carry much of the effect (the top 4 principal directions remove 77 %). Second, the "mass read-out
does not move" statement holds for the mean over jets, but on single jets the read-out moves by 0.04 in $\log m$ on average.

## Files
- Script `interp/e26_verify.py`. Smoke run `e26-vsmoke` (500 jets, exit 0). Full run:
  `./job.sh e26-verify 'python -m interp.e26_verify --seed 2602'` (1 CPU / 2 GiB, wall 565 s, exit 0, 0.52 cores). The pod peak
  memory reads 2047 MiB on a 2 GiB request, which is the cgroup limit including page cache (the same as E26's smoke); the run was not killed.
- Results: `results/jedi/e26_verify.json` (and `e26_verify_smoke.json`). Logs: `logs/e26-verify.log`, `logs/e26-vsmoke.log`.

## Code review of `interp/e26_boost_route.py` and `interp/e26_plot.py`
1. **Random-subspace null (main run and `--null1`) is drawn in all 64 S coordinates, but only 19 S units are active.**
   `X.rand_Q(64, d, rng)` spreads each direction over the 45 dead coordinates. The projection $QQ^\top\Delta S$ then puts
   0.690 of its squared norm on dead coordinates (measured, 200 draws, $\lambda$ 0.9 and 1.1). The fused d4 weights from those coordinates
   are exactly 0 (row norm 0.0 vs 3.79 for live rows). The clip range `rng_['S']` is the quantizer's representable range, not
   the data range, so it does not remove this. A random "d-dim" hold is therefore about a 0.3·d-dim hold, and the null is too lenient.
   The DAS directions put 0.9995 of their weight on live coordinates, so they are not comparable in size.
2. **Holm over 130 single units uses one null for all stages.** The null is random S directions (item 1), but it is applied to
   l1, g, h4 and h5 units as well. For the four h4/h5 units reported at $\lambda$ 1.1 (and h4 53, h5 24 at 0.9), the matched
   null is random live-h4 / live-h5 directions. Those h4 / h5 results were not re-tested here.
3. **Figure** (`results/jedi/e26_boost_route.png`, read): the shaded band is the 1-dim, 64-coordinate random null. It sits under
   bars of 1-, 2-, 4- and 13-unit holds, so the multi-unit bars have no size-matched band. The suptitle ("runs through ΣpT …,
   not through the mass read-out") is an interpretive statement. The right panel matches the json. Axes are labelled.
4. Checked and found consistent: padded slots stay zero through `boost`, `pair_inputs` and `hls_prep` (pT = 0 slots are sorted
   last in both the boosted and the unboosted-in-boosted-order inputs). The l1 hold overrides `a` after l1, so P, s and u are
   recomputed. The exact tail reproduces `run_jetwise` (max |Δlogit| 0, re-checked here). The ridge read-outs are fit on half A
   and read on half B. B keeps $m$ fixed to 1e-11. The DAS fit sample (`sample(4000, 0)`) is drawn from the same test file as the
   evaluation sample. Overlap is possible, but the DAS fit uses interchange labels, not evaluation call changes.

## Procedure (this verifier)
- **Sample and network:** `interp.jedi_core.sample(2000, seed 2602)`, 10,000 jets; official 3-feature n64; unedited accuracy 0.8066.
- **Edits and holds:** the E26 code paths (`boost`, `pair_inputs`, `clamp_pt`, `Ctx`, `Net`; imported). DAS M0 / M2 directions
  at S were refit with E26's procedure (X10 data, pairs and seeds). Validation balanced agreement: M0 0.90 / 0.91, M2 0.70 / 0.76,
  the same as E26.
- **Metric:** share of B's call changes removed, with a 95 % Poisson bootstrap over jets (1,000 draws) and McNemar.
- **New controls** (all evaluated on half B, 5,000 jets):
  - (a) Random directions in all 64 S coordinates (E26's null): 2,000 draws at dim 1, 500 at dims 2 and 4.
  - (b) Random directions restricted to the 19 live S coordinates, with the same draw counts.
  - (c) The top-k principal directions of unedited S (fit on half A, centred). This subspace contains no boost or mass information.
  - (d) The top-k right singular vectors of $S(B)-S(0)$ (fit on half A). This is an upper bound for a linear hold of that dimension.
- **Read-out check:** the E15 ridge read-out of $\log m$ at h4 (fit on half A unedited, held-out $R^2$ 0.973, residual sd 0.086).
  - The per-jet distribution of $\Delta\log\hat m$ under B.
  - A causal test of the 1-dim read-out direction $q \propto w/\sigma$ in h4. "Hold" puts B's component along $q$ back to its
    unboosted value. "Patch" puts only B's component along $q$ onto the unboosted h4.
  - Both are compared with 300 random 1-dim directions in the 55 live h4 coordinates.

## Results

### Replication (seed 2602 vs E26 seed 2601), removed share [95 % CI]
| hold | λ 0.9 verify | λ 0.9 E26 | λ 1.1 verify | λ 1.1 E26 |
|---|---|---|---|---|
| B calls changed | 32.3 % [31.4, 33.2] | 32.0 % | 14.5 % [13.8, 15.2] | 14.6 % |
| all 13 live l1 cuts | 99.3 % [99.0, 99.5] | 99.0 % | 98.8 % | 98.6 % |
| whole S | 100 % | 100 % | 100 % | 100 % |
| DAS M0+M2 at S, 2 dims | 85.0 % [83.7, 86.3] | 83.2 % | 51.1 % [47.2, 54.9] | 55.3 % |
| DAS M0+M2 at S, 4 dims | 84.8 % [83.5, 86.1] | 84.8 % | 62.8 % [59.3, 66.0] | 64.9 % |
| DAS M0 at S, 1 dim | 62.0 % [60.1, 63.8] | 60.2 % | 5.1 % [−1.2, 10.9] | 3.6 % |
| DAS M2 at S, 1 dim | 14.1 % [12.2, 16.0] | 16.5 % | −43.5 % [−49.8, −37.8] | −43.2 % |
| S0 | 56.8 % [54.8, 58.7] | 55.1 % | −0.6 % [−7.0, 5.5] | −0.3 % |
| S0 + S29 | 78.9 % [77.3, 80.3] | 78.2 % | 51.3 % [47.8, 54.9] | 50.4 % |
| S0 + S19 | 56.8 % | 56.1 % | 0.1 % | 0.7 % |
| g49 | 50.7 % [48.8, 52.7] | 50.5 % | 7.5 % [2.9, 12.2] | 7.8 % |
| g (all) | 54.1 % | 55.0 % | 13.4 % | 12.7 % |
| a15 | 41.6 % [39.6, 43.7] | 39.9 % | −24.8 % | −26.0 % |
| 4 pT-ramp cuts | 14.5 % [11.9, 17.0] | 12.5 % | −61.7 % | −64.0 % |
| a32 | −1.9 % | −2.0 % | −0.5 % | 0.1 % |
| Bc (11 GeV crossings clamped) | 0.3 % [−0.0, 0.7], McNemar p 0.11 | −0.2 % | 0.2 %, p 0.55 | 0.2 % |

The net removed share hides changes the hold creates. Discordant counts (removed / created):
- S0 at λ 0.9: 2,125 / 290.
- S0 at λ 1.1: 1,006 / 1,014.
- 4 pT-ramp cuts at λ 1.1: 947 / 1,841.

### Null adequacy (half B, 5,000 jets; B changes 32.4 % at λ 0.9, 14.7 % at λ 1.1)
Removed share, 97.5 % quantile / maximum of the random nulls:

| null | λ 0.9, 1 dim | λ 0.9, 2 dims | λ 0.9, 4 dims | λ 1.1, 1 dim | λ 1.1, 2 dims | λ 1.1, 4 dims |
|---|---|---|---|---|---|---|
| random, 64 coordinates (E26) | 0.205 / 0.440 | 0.271 / 0.422 | 0.326 / 0.401 | 0.107 / 0.199 | 0.120 / 0.170 | 0.117 / 0.179 |
| random, 19 live coordinates | 0.324 / 0.482 | 0.335 / 0.448 | 0.389 / 0.563 | 0.087 / 0.216 | 0.004 / 0.104 | −0.003 / 0.137 |

Non-random subspaces of the same dimension:

| subspace | λ 0.9, 1 dim | λ 0.9, 2 dims | λ 0.9, 4 dims | λ 1.1, 1 dim | λ 1.1, 2 dims | λ 1.1, 4 dims |
|---|---|---|---|---|---|---|
| DAS (M0 at 1 dim; M0+M2 at 2 and 4 dims) | 0.620 | 0.845 | 0.854 | 0.049 | 0.532 | 0.672 |
| unit axes (S0; S0+S29) | 0.566 | 0.790 | – | −0.008 | 0.537 | – |
| top principal directions of unedited S | 0.011 | 0.467 | 0.774 | −1.271 | −0.065 | 0.494 |
| top singular directions of $S(B)-S(0)$ (bound) | 0.799 | 0.853 | 0.860 | 0.601 | 0.588 | 0.590 |

Tests against the live-coordinate null:
- **λ 0.9:** $p_\text{rand}$ is at the floor of the draw count for DAS M0+M2 (2 and 4 dims: 0.002), S0 (0.0005) and S0+S29 (0.002).
  S0 with Holm over 130 units gives a bound of 0.065. This bound is limited by the 2,000 draws; S0's 0.566 is above the live-null
  maximum of 0.482.
- **λ 1.1:** DAS M0+M2 has p 0.002 and S0+S29 has p 0.002. S0 has p 0.17.

Overlap (mean squared cosine) of the DAS M0+M2 2-dim subspace:
- with the top-2 principal directions of S: 0.55;
- with the top-2 singular directions of $S(B)-S(0)$: 0.50;
- with the S0 and S29 axes: 0.58.

### h4 mass read-out under B (half B)
| | λ 0.9 | λ 1.1 |
|---|---|---|
| mean $\Delta\log\hat m$ (B) | −0.0034 | −0.0043 |
| per-jet sd of $\Delta\log\hat m$ (B) | 0.049 | 0.046 |
| mean $\lvert\Delta\log\hat m\rvert$: changed calls / unchanged calls | 0.034 / 0.040 | 0.035 / 0.036 |
| mean $\Delta\log\hat m$ under P (true $\log\lambda$) | −0.110 (−0.105), sd 0.037 | +0.089 (+0.095), sd 0.034 |
| hold of the read-out direction: removed share | −0.3 % (null 97.5 % 9.6 %, p 0.53) | 1.4 % (null 97.5 % 12.4 %, p 0.59) |
| patch of B's read-out component onto unboosted h4: calls changed | 0.8 % (null median 1.6 %) | 0.6 % (null median 1.3 %) |

## Hypotheses
- **H0 (random-hold null):** a named hold removes no more than a random hold of the same dimension.
  - At λ 0.9, H0 is rejected for S0, S0+S29 and DAS M0+M2 against both random nulls (p at the draw floor).
  - At λ 1.1, H0 is rejected for S0+S29 and DAS M0+M2 (p 0.002), and not rejected for S0 (p 0.17).
- **H1 (DAS M0/M2 directions are specific):** the DAS M0+M2 subspace removes more than generic structured subspaces of S.
  - At 2 dims it does: 0.85 vs 0.47 for the top-2 principal directions, and it reaches the 2-dim bound of 0.85.
  - At 4 dims the principal directions reach 0.77 vs 0.85, so the margin is smaller.
  - The DAS subspace was fit on M0/M2 interchanges, not on the boost. Its match to the bound is consistent with the boost acting at
    S mainly through the M0 and M2 directions.
- **Mass read-out route:** "the call change is carried by the h4 read-out of $\log m$". Excluded causally for the 1-dim ridge
  direction: holding it removes −0.3 % (λ 0.9). The read-out's per-jet movement is not larger on jets whose call changes.
  - Not tested: other h4 directions that also encode mass.

## Corrected reading
- **λ 0.9:** holding S0 removes 55–57 % of the boost's call changes (two samples). Holding S0+S29 removes 78–79 %, and holding the
  DAS M0+M2 directions at S removes 83–85 %.
  - Random directions restricted to the live S coordinates reach at most 0.48 (1 dim), 0.45 (2 dims) and 0.56 (4 dims). E26's
    0.43–0.48 came from a null diluted by dead coordinates.
  - The top 4 principal directions of S remove 77 %.
- **Mass read-out:** the h4 read-out does not move on average ($|\overline{\Delta}| \le 0.008$), but it moves per jet (sd 0.05).
  Holding its direction removes no call changes. "Equal weights" for M0 and M2 refers to the mean response.
- **λ 1.1:** M0 and M2 act in opposite senses. DAS M2 alone adds 43 % more changes, and M0+M2 together remove 51–55 %.
  - S0 alone removes about 0 % (p 0.17 vs the live null).
  - The h4/h5 single-unit Holm results at λ 1.1 used an S-level null and were not re-tested with a matched h4/h5 null.
- The rest of E26's reading repeats on the new sample within the intervals: the pT ramps, the 11 GeV threshold (a32 hold, Bc clamp),
  g49 and the λ > 1 behaviour.

## Explainer entry (for site/explain.py, Paper A tab, under E26)
- **Question:** are E26's boost-route numbers stable on new jets, and was the random-direction comparison fair?
- **Background:** a "hold" resets part of the network's internal state to its value on the unboosted jet. The fraction of
  call changes that disappear measures how much of the boost response passes through that part. A fair comparison uses random
  holds of the same effective size. Directions in coordinates the next layer ignores (zero weights) are not effective.
- **Method:** repeat the key holds on `sample(2000, 2602)`. Compare them with random directions drawn only in the 19 active
  coordinates of S, with the top principal directions of S, and with the best possible $k$-dim subspace (top singular vectors
  of $S(B)-S(0)$). Test the h4 mass read-out direction $q\propto w/\sigma$ causally by holding and patching it.
- **Result:** the replication table above. At $\lambda=0.9$, S0 removes 57 % and the $M_0$/$M_2$ directions remove 85 %. Random
  live directions reach at most 48 % (1 dim) and 56 % (4 dims).
- **Checks:** the exact-tail identity (max $|\Delta\text{logit}|$ 0); holding all cuts removes 99.3 %; DAS refit agreement equals E26's.
- **Why it matters:** the claim that the boost response runs through $M_0=\sum p_T$ and $M_2=\sum p_T r^2$ at S holds against a
  stricter null. The size of the random baseline was understated, and generic high-variance directions carry part of the effect.

## Conclusion (site/conclusions.py, key-result form)
The E26 boost result repeats on new jets: holding the summed-pT object, or the two learned ΣpT and angular-spread directions, at
their pre-boost values removes about 57 % and 85 % of the changed calls at 10 % lower jet pT. A stricter random control (random
directions only in the active coordinates) still falls below both. Generic high-variance directions also carry part of the
effect. The network's mass read-out stays fixed on average but moves on single jets; holding it removes none of the changes.
