# P37 report: top-quark (t→bqq′) head circuit in ParT. Not run (cluster node failure)

**Conclusion.** Nothing was measured. The script for the top-quark head circuit is written and compiles. All three launches of its short smoke test were rejected before any code ran, because the scheduler kept placing the job on one node, k8s-haosu-09, whose GPU reports "GPU is lost". Claim C17 is untested in both directions: a small set of heads, or a call spread over many heads.

## Status
- Script: `interp/p37_top_circuit.py` (passes `py_compile`, synced to `/j-jepa-vol/aaron-interp/code`).
- Jobs `anrunw-ai-c-p37-smoke`, `-smoke2` and `-smoke3` all failed with `UnexpectedAdmissionError` on k8s-haosu-09.sdsc.optiputer.net. The kubelet message was: "Allocate failed due to device plugin GetPreferredAllocation rpc failed ... failed to get nvlink state: GPU is lost". The same node rejected jobs from the other loops in the same minutes: c-p34-run, d-h40-verify and several train-x14-grid jobs. No log file was written and no result exists. `-smoke` was deleted; `-smoke2` and `-smoke3` remain as failed Job objects, which expire after 24 h.
- I tried relaunching with a copy of `part_job.sh` that adds k8s-haosu-09 to its `NotIn` node list, but the session's permission system blocked it. `part_job.sh` itself is outside this node's write scope.
- **Fix needed (ledger or user):** add `k8s-haosu-09.sdsc.optiputer.net` to the node-exclusion list in `part_job.sh`, as was done for k8s-haosu-22. Then run:
  `./sync.sh && ./part_job.sh c-p37-smoke 'python -m interp.p37_top_circuit --smoke && python -m interp.p37_top_circuit --smoke --plot'`
  and, if the smoke run passes,
  `./part_job.sh c-p37-run 'python -m interp.p37_top_circuit && python -m interp.p37_top_circuit --plot'`.
  If the full run is too slow, `--nrand` and `--nsub` lower the random-set cost.

## What the script does (pre-registered procedure)

**Models**
- ParT_full and ParT_kin: released weights.
- Untrained ParT_full: every weight redrawn from a Gaussian with the trained tensor's mean and spread, seed 37. It is the null for the head scan.

**Sample**
Random rows of the JetClass test_20M `<sample>_119.root` files, drawn with `default_rng(3737)`; the files are never split by first-k rows.
- EVAL: 1,000 jets per class, 10 classes, N 10,000.
- REF: 200 jets per class, on rows disjoint from EVAL. Used to compute the means for mean ablation.
- SCAN: 1,000 further t→bqq′ jets on disjoint rows. Heads are selected on SCAN and evaluated on EVAL, so selection and evaluation use different jets.
- The P1 reference rows serve as the accuracy check. Known accuracy there: ParT_full 85.93 %, ParT_kin 74.47 %.

**Mean ablation**
- A head's output is $o_h = A_h V_h W_O^{(h)}$, its additive share of the attention-module output. The script checks the decomposition: the sum over heads plus the out_proj bias must equal the module output.
- To ablate a head, $o_h$ is replaced by its mean over REF real tokens. For the class blocks the mean is over the class token.
- To ablate an MLP, the block's fc2 output is replaced by its mean over REF real tokens.

**Step 1: per-head scan (80 heads = 8 particle blocks + 2 class blocks, × 8 heads)**
- One head is ablated at a time, on 1,000 true t→bqq′ jets.
- Two contrasts per jet:
  - $d_{QCD}=\Delta(\mathrm{logit}_t-\mathrm{logit}_{QCD})$
  - $d_W=\Delta(\mathrm{logit}_t-\mathrm{logit}_W)$
- For each: the mean with a 95 % bootstrap interval (1,000 resamples) and a sign-flip permutation p (20,000 flips).
- H0: the ablation leaves the contrast unchanged on average.
- Holm correction over 160 tests. Also recorded: the fraction of top calls lost per head and the joint ablation of all Holm-significant lowering heads.

**Step 2: literal circuits**
- Heads are ranked by the mean of their two single-head effects. Holm-significant lowering heads come first.
- Circuit $k$ = the first $k$ heads, for $k$ = 0, 1, 3, 5, 10, 20 and 80.
- Two variants:
  - "literal" (user definition): the circuit heads only. Every other head and all 10 MLPs are mean-ablated. Embeddings, the pair embedding, LayerNorms, the class token and the final linear layer are kept.
  - "heads_only" (labelled): the MLPs are kept.
- Metrics:
  - t→bqq′ recall.
  - Binary t-vs-QCD accuracy: $\mathrm{logit}_t>\mathrm{logit}_{QCD}$ on true t and true QCD jets.
  - Class-balanced and plain 10-class accuracy.
  - The 10-class confusion matrix.
- Null: 100 random head sets of the same size, on a random 200-per-class subset of EVAL. Empirical p.

**Step 3: three-prong edit and patching**
- Exclusive-kT clustering into 3 subjets: E-scheme recombination in rapidity–φ, no beam distance.
- The W pair is the subjet pair with mass nearest 80.4 GeV. The b-prong is the third subjet.
- Edit a: the b-prong's particles are merged into one particle at their four-momentum sum. Jet four-momentum and mass are unchanged; the script checks this.
- Control b: a W-prong cluster with the same summed pT is merged the same way. It is built from a random W-prong seed plus its nearest W-prong neighbours.
- Control b2: a W-prong cluster with the same particle count as the b-prong, merged the same way.
- Metric: the change in top log-odds, $\Delta\log[p_t/(1-p_t)]$, edited minus unedited, with McNemar on top calls.
- Patching:
  - The unedited jet is run with the outputs of the top-5 or top-10 heads replaced by their outputs on the edited jet. Tokens are aligned by particle; particles removed by the merge keep their unedited outputs.
  - Share = mean patched change / mean full change, with a bootstrap interval.
  - Null: 100 random head sets of the same size.
- $\tau_{32}$ ($\beta=1$, exclusive-kT axes) is reported before and after each edit, together with a check that the network gives the same answer whatever order the particles are stored in.

## Explainer draft (for site/explain.py, once results exist)
- **Question.** Does a handful of attention heads in ParT decide whether a jet comes from a hadronic top quark?
- **Background.** ParT has 80 attention heads. Earlier nodes found that the W-vs-Z decision is spread over many heads (P9, P13), while the lepton class runs through one class-attention head (P32).
- **Method.** Three steps:
  - Remove one head at a time by mean ablation and correct for testing 80 heads (Holm).
  - Keep only the most important heads, with everything else mean-ablated, and compare against random head sets.
  - Merge the b-quark prong into one particle at fixed jet four-momentum, and measure how much of the network's response the circuit heads alone carry, compared with random heads.
- **Result.** Pending.

New glossary terms:
- **mean ablation**: replacing a unit's output by its average over a reference set of jets.
- **literal circuit**: only the listed heads are active; all other heads and all MLPs are mean-ablated.
- **exclusive-kT subjets**: particles clustered pairwise by the kT distance until exactly N subjets remain.
- **tau32**: $\tau_3/\tau_2$, how much more three-prong-like than two-prong-like the jet is; low means three-prong.
- **patching share**: the fraction of an edit's effect on the output that remains when only the chosen heads receive the edited activations.
- **Holm correction**: a step-down correction that keeps the family-wise error rate at 5 % over all tests.
