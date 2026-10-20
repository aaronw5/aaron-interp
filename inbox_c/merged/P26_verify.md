# P26 verify: Sophon axis response, rerun on fresh rows plus three added controls

Status: done. Job `anrunw-ai-c-p26-verify` (exit 0, 440 s). Script `interp/p26v_verify.py`. It imports `interp/p26_sophon_axis.py` unchanged for samples, edits, models and `analyse`; only the JetClass-II shard list is a parameter. Results are in `results/part/p26_verify_run.json` (volume: `$AI_RESULTS/part/p26_verify_run.json`).

## Conclusion

Conclusion: On new jets, Sophon again shows almost no W/Z output change under a real 12 % mass increase. Its internal mass read-out again moves only about 0.2 as much under the axis shift as under a real mass change, against about 0.8–1.0 for the JetClass ParT. A probe-free check agrees and is stronger: the axis shift moves Sophon's class token slightly against the real-mass direction (ratio −0.3 to −0.1), not along it. The worker's phrase "its mass representation is mostly invariant" needs narrowing. About half of Sophon's response to a real size change enters through the axis-relative angular inputs, so Sophon is not reading only the four-vector invariant mass. What it does is read the jet's angular size in a way that mostly ignores where the axis sits. Its output does still change under the axis edits, in directions unrelated to mass, and why it ended up this way (data scale, mass decorrelation or pT-normalised inputs) is still open.

## Code review (interp/p26_sophon_axis.py, interp/sophon_part.py, interp/p26_plot.py)

- **Class map.** Checked against the released `data/JetClassII/JetClassII_full.yaml` (GitHub, jet-universe/sophon). Labels 0–8 are X→bb, cc, ss, qq, bc, cs, bq, cq, sq and labels 161–187 are the 27 QCD classes, so `GROUP_Z`/`GROUP_W`/`GROUP_QCD` are correct. Feature order, centre/scale/clip, `part_*_scale` = p·500/jet_pt, wrap padding and length 128 also match the yaml. The training reweighting is flat in jet_pt (200–2500) and jet_sdmass (20–500, 10 GeV bins) per reweight class, and all nine X→QQ labels form one reweight class. This is the mass decorrelation the worker cites.
- **Hooks.** The class token is the input of `fc` (after the final LayerNorm), for both Sophon and ParT. ParT's angular channels are ΔR 4, Δη 15, Δφ 16 (full) and 4, 5, 6 (kin). Sophon's are 4, 15, 16, as in `SOPHON_IDX`.
- **Sampling.** Sample A uses `rng.choice` on random rows and B uses random candidates, so no first-k rows are taken from class-ordered files. The build uses η = jet_eta + sign(jet_eta)·Δη, and `reaxis` uses the same convention.
- **Edits.** φ+0.06 changes only Δφ and ΔR; the four-vectors rotate rigidly, so pair features are unchanged (Δlog m_inv ≈ 10⁻⁸). The dilation gives Δlog m_inv 0.122 and Δlog m_A 0.126. The ΔR-keeping rotation changes Δlog m_inv by −0.001 (A) and −0.009 (B). No edit changes more than stated.
- **Masking.** Padded slots are masked: `mask` comes from pt > 0, wrap-padded values sit behind mask 0, and the ONNX check agrees to 6·10⁻⁶.
- **Not checked.** The yaml's `part_d0`/`part_dz` definition (the script uses tanh(d0val), tanh(dzval)) and the d0/dz units of JetClass versus JetClass-II. These only affect the full-feature inputs, and they belong to the domain caveat the worker already raised.
- No bug found that changes a reported number.

## Rerun on fresh rows

Samples:
- **A′**: JetClass test_20M `{ZJetsToNuNu,ZToQQ,WToQQ}_113.root` (P26 used `_112`), 5,000 random rows per class, `default_rng(2699)`, N 15,000.
- **B′**: JetClass-II `Res2P_0254–0258` and `QCD_0351` (P26 used 0250–0253 and 0350), same selection, `default_rng(2698)`, N 6,000 (2,000 per group).

Analysis seed 2699; bootstrap 2,000 over jets.

| quantity | P26 (A / B) | verify (A′ / B′) |
|---|---|---|
| Sophon mapped 3-class acc | 0.626 / 0.752 | 0.622 / 0.737 |
| Sophon Δd on W, dilation, / Z−W separation | 0.016 / −0.004 | 0.018 / −0.001 (Δd +0.035 [0.030, 0.040] / −0.002 [−0.013, 0.008]) |
| Sophon Δd on W, φ+0.06 | −0.097 / −0.069 | −0.093 [−0.104, −0.083] / −0.066 [−0.078, −0.054] |
| Sophon R (output) | −0.256 / −0.006 | −0.322 [−0.369, −0.277] / −0.026 [−0.079, 0.024] |
| Sophon R_probe | 0.125 / 0.252 | 0.179 [0.163, 0.197] / 0.232 [0.211, 0.254] |
| Sophon probe R² | 0.76 / 0.78 | 0.79 / 0.75 |
| Sophon W→Z under φ | 24.3→23.5 % / 20.9→19.7 % | 25.4→23.8 % (McNemar p 1.5·10⁻⁷) / 20.2→19.1 % (p 0.006) |
| Sophon top-1 changed: WTA / translation / rotation | 7.3 / 6.6 / 16.0 % (A) | 7.2 / 6.7 / 16.6 % (A′; WTA vs translation p 0.029); 6.9 / 6.7 / 17.7 % (B′; p 0.50) |
| ParT_full R / R_probe | 0.801 / 0.776 (A) | 0.792 [0.781, 0.803] / 0.797 [0.778, 0.817] (A′); 0.774 / 0.980 (B′) |
| ParT_full W→Z under φ | 11.5→32.9 % | 11.5→31.9 % |
| ParT_kin R / R_probe (A) | 0.734 / 1.507 | 0.738 / 1.456 |
| untrained (seed 100) R_probe | 0.844 | 0.829 [0.803, 0.855] |

All P26 numbers reproduce within about 0.07 on fresh rows. One difference: on B′, WTA versus the same-length random axis move is no longer different (p 0.50; P26 B p 0.036).

## Added controls

Each control uses the same jets as the rerun. The class token is h, the 128-d input of the final classifier.

### 1. Probe calibration against the real change

Mean Δprobe(dilation) / mean Δlog m_inv(dilation), test half (95 % bootstrap):

| model | A′ | B′ |
|---|---|---|
| Sophon | 0.46 [0.45, 0.48] | 0.56 [0.54, 0.57] |
| ParT_full | 0.72 | 0.76 |
| ParT_kin | 0.73 | – |
| untrained seed 100 | 0.15 (R² 0.26) | – |

The probe recovers about half of Sophon's real mass change. The untrained network's probe recovers 15 %, so its R_probe (0.56–0.84) is a ratio of small shifts and is a weak reference.

### 2. Probe-free class-token ratio

Let u be the unit vector of the mean Δh under dilation. R_h = mean(Δh(φ)·u) / mean(Δh(dil)·u). H0: R_h(Sophon) = R_h(other model), tested with a paired bootstrap on the same jets and the same weights.

| model | R_h, A′ | R_h, B′ | cos(mean Δh(φ), mean Δh(dil)), A′ / B′ |
|---|---|---|---|
| Sophon | −0.298 [−0.310, −0.286] | −0.121 [−0.134, −0.108] | −0.39 / −0.32 |
| ParT_full | 0.843 [0.833, 0.852] | 0.894 [0.878, 0.912] | 0.98 / 0.94 |
| ParT_kin | 0.889 | – | – |
| untrained seed 100 | 0.424 | – | – |

- Sophon − ParT_full: −1.14 [−1.16, −1.12] (A′) and −1.02 [−1.04, −1.00] (B′).
- R_probe differences: Sophon − ParT_full −0.62 [−0.64, −0.60] (A′) and −0.75 [−0.78, −0.71] (B′); Sophon − ParT_kin −1.28; Sophon − untrained −0.65.
- All 8 tests have p ≤ 0.001, the floor at 2,000 resamples. Holm over the 8 tests gives 0.008.
- This closes P26's "no formal test between probe ratios".
- Sophon's sign is negative: the φ shift moves the token slightly against the real-size direction. An axis-moment reader predicts R_h ≈ 1 and an axis-blind reader predicts R_h ≈ 0. The axis-moment prediction is excluded for Sophon.

### 3. Route split of the dilation response

Two partial edits separate the routes:
- **dil_vec**: dilated four-vectors with the per-particle Δη, Δφ, ΔR fed unedited. The pair features and pT/E features move; the axis-relative inputs do not.
- **dil_ang**: unedited four-vectors with the dilated Δη, Δφ, ΔR. Only the axis-relative inputs move.

Share of the full-dilation response (probe on the test half; token projection on u):

| model | probe dil_vec / dil_ang | token projection dil_vec / dil_ang |
|---|---|---|
| Sophon A′ | 0.40 / 0.57 | 0.69 / 0.29 |
| Sophon B′ | 0.47 / 0.50 | 0.56 / 0.41 |
| ParT_full A′ | 0.05 / 0.95 | 0.19 / 0.81 |
| ParT_full B′ | 0.05 / 0.95 | 0.21 / 0.79 |
| ParT_kin A′ | 0.18 / 0.87 | 0.14 / 0.83 |
| untrained A′ | 0.00 / 1.00 | 0.01 / 1.00 |

- In ParT_full, nearly the whole mass read-out runs through the angular per-particle inputs. In Sophon, 40–69 % runs through the four-vectors (pair features), and the rest still runs through the angular inputs.
- The φ shift edits only the angular inputs. In Sophon it gives 0.18–0.23 of the full dilation probe response. That is 0.32 (A′) and 0.47 (B′) of the angular-only dilation, so Sophon's angular route is partly, not fully, insensitive to the axis offset.
- Prediction of a pure invariant-mass (four-vector) reader: dil_ang share 0. That prediction is excluded for Sophon (shares 0.50–0.57, intervals ±0.01).
- Prediction of a reader of the centroid-relative angular spread M0·M2 − |M1|²: dil_ang share > 0 and φ response ≈ 0. This spread is unchanged by a rigid shift. The data fit this better than either the four-vector reader or the axis-moment reader, but no edit separated it directly.

### 4. Size of the token response to axis edits (rms |Δh|, Sophon A′)

| edit | rms \|Δh\| |
|---|---|
| φ+0.06 | 0.048 |
| dilation | 0.065 |
| WTA re-axis | 0.065 |
| same-length random axis move | 0.072 |
| ΔR-keeping rotation | 0.105 |

The rotation moves 16–18 % of Sophon's top-1 calls (ParT_full 8–11 %). The φ shift moves Sophon's W/Z read-out 2.6× as much as the dilation does (−0.093 vs +0.035 on W jets). So Sophon's token and output do respond to axis-relative edits. The response is in directions other than its mass direction.

## Adequacy of P26's controls

- **Rotation null** (rot, |R| ≤ 0.016) and the matched dilation: adequate.
- **ΔR-keeping rotation as the WTA control**: weak for Sophon. Sophon is more sensitive to particle orientation about the axis than to the WTA re-axis itself, so "WTA below the rotation control" does not show insensitivity to the axis. The same-length random axis move is the relevant control: Sophon's WTA ≈ that control (A′ 7.2 vs 6.7 %, B′ 6.9 vs 6.7 %), while ParT's WTA is below it (21.5 vs 27.8 %).
- **Untrained reference**: its probe has R² 0.26 and calibration 0.15. "0.56–0.84 for an untrained copy" should be quoted with that caveat. The route split shows the untrained token is 100 % angular-route, which is the cleaner statement.

## Figure (results/part/p26_sophon_axis.png)

- The panels match the JSON.
- Panel (b) labels R_probe = 0 "reads the invariant mass". Control 3 shows that R_probe ≈ 0 is also produced by a shift-invariant angular reading, so "insensitive to the axis offset" is the measured meaning.
- The suptitle ("a probe that follows the invariant mass more than ParT's") carries the same overstatement.
- Panel (c) would read better with the same-length random axis move as the named control.

## Corrected reading

1. **Output.** Sophon's flavour-group W/Z output barely changes under a real 12 % mass increase: 1.6–1.8 % of the Z−W separation on JetClass and ≈ 0 on JetClass-II, reproduced on fresh rows. R on the output is a ratio of two small numbers.
2. **Class token.** Its log-mass probe (R² 0.75–0.79, calibration 0.46–0.56) moves under the φ shift at 0.13–0.25 of its dilation movement. This is below ParT_full (0.78–0.98) and the untrained architecture (0.83), with paired bootstrap p ≤ 0.001 (Holm 0.008). The probe-free projection gives R_h −0.30 / −0.12 against ParT_full 0.84 / 0.89.
3. **Routes.** About half of Sophon's real-size response enters through the per-particle angular inputs. Its size reading is therefore largely insensitive to the axis position; it is not a pure four-vector invariant-mass reading.
4. **WTA re-axing.** It changes Sophon's calls at the level of a same-length random axis move (ParT: below that control). The ΔR-keeping rotation changes more Sophon calls than either.
5. **Confounds.** Data scale, mass decorrelation and pT-normalised four-vectors remain confounded.

## Statistics summary

- H0: Sophon's class token responds to the axis shift as to a matched real size change (R_h = 1, the C3 reading). H1: R_h < 1.
  - Result: R_h −0.30 [−0.31, −0.29] (A′) and −0.12 [−0.13, −0.11] (B′); H0 excluded.
  - Control: ParT_full on the same jets, 0.84 / 0.89.
- Probe ratios use the paired bootstrap above, Holm over 8 tests.
- W→Z changes use McNemar (A′ p 1.5·10⁻⁷, B′ p 0.006).
- Route shares come with 95 % bootstrap intervals over jets (2,000).

## Explainer addendum (for site/explain.py, P26 entry "Checks")

- **Fresh rows**: JetClass `_113` and JetClass-II shards 0254–0258 reproduce every P26 number within 0.07.
- **Probe-free check**: with $u=\overline{\Delta h_{\rm dil}}/|\overline{\Delta h_{\rm dil}}|$, $R_h=\overline{\Delta h_\phi\cdot u}/\overline{\Delta h_{\rm dil}\cdot u}$ is $-0.30$ (Sophon) against $0.84$ (ParT_full).
- **Route split**: feeding the dilated four-vectors with unedited $(\Delta\eta,\Delta\phi,\Delta R)$ carries 40–47 % of Sophon's probe response (ParT_full 5 %). Sophon reads jet size partly from the four-vectors and partly from the angular inputs, in a way that a rigid shift of the axis barely affects; a candidate is the spread about the $p_T$ centroid, $M_0M_2-|M_1|^2$.
