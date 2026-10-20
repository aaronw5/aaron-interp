# AGENTS.md: full-context handbook for any agent working in this repo

Read this first. Then read, in order: `PAPERS.md` (theses, claims, claim confidence, narrative, reviewer lens),
`EXPERIMENTS.md` and `TREE.md` (every experiment and result), `WORKFLOW.md` (loop structure, statistics, documents),
`CLUSTER.md` (NRP rules), `QUESTIONS.md` (the user's answers and open questions), `RESUME.md` (how to resume paused work).
Last full update: 2026-10-19.

## 1. Goal

The project interprets jet taggers with formulas "from front to back", for mechanistic-interpretability, method, circuits and
physics papers. The user's working thesis: "the model creates classification boundaries by making different objects off of
statements"; also "high level variables are built over layers". Order: JEDI-linear first, then small models trained on hls4ml,
then big models (ParT, ParticleNet, L-GATr). Target venue: ICML or another top venue. The deadline is about three months after
2026-10-17, so mid-January 2027; the exact date has not been confirmed.

## 2. User instructions (binding; dated; quoted where the wording matters)

### Research and papers
- 2026-10-16: "Make sure you evolve or keep same your thesis hypothesis for each paper as you go. Spawn agents to do
  implementations for you and report back to you. Keep track of experiments and what they tell you. Experiments should help the
  paper NARRATIVE." Theses change only with a dated entry; old wording is kept with the node id.
- 2026-10-16: "Again what's most important is a compelling narrative for the papers." Think about how each paper will be reviewed.
- 2026-10-16: "I don't think it can JUST be trigger tagger right? There should be at least an attempt on bigger model." Also: "If you
  feel like we can get a FULL STRONG PAPER off the toy or small models that's also very valuable."
- 2026-10-16: "Just use the best version of Jedi." (Working model: 3-feature n64; whether to switch to 16-feature-perminv n64 is
  an open question.) "For Jedi u don't need to look at ALL of them."
- 2026-10-16: the MARS fit (turning a model into if-statements) should be written about or explored ("Or explore it"); fit the
  ENTIRE trigger model, check it reproduces the performance, see how features evolve and what each neuron looks at, and how close
  that is to physics.
- 2026-10-16: "Also similar to the good fire post I sent, u should look at neurons too" (Goodfire, "Neural Networks Are Already Clean",
  https://already-clean.goodfire.com).
- 2026-10-16: "If papers become too similar or fail to be standalone they should be combined."
- 2026-10-16: one science-loop workflow per paper, "max 3" loops at once.
- 2026-10-16: models: "lgatr and particle net", "Big ones"; "For small models you can train native transformer or Linformer on
  hls4ml". Datasets added: Top Tagging Landscape ("only 1 class") and Quark–Gluon.
- 2026-10-16: "Always check if there are weights available for the task online first." For L-GATr: "Train own, or use their
  checkpoint stored in their git." Never use other groups' run checkpoints from the volume.
- 2026-10-17: "Don't freeze paper right we should explore more" and "we have 3 months until deadline we can explore": exploration
  phase. No freezing of theses, no confirmatory-only rounds yet; at least 2 of 3 loop nodes per round are exploratory.
- 2026-10-19: "I mean it doesn't need to be a circuit paper… if there's no circuit lol keep exploring we don't need to think of an
  ENTIRE PAPER yet." Follow what the models do; negative findings are results; do not shape rounds around a finished paper.
- 2026-10-17: "Yes start more loops": three loops (A/B, C, D; section 4).

### Write-up gate
- 2026-10-16: "Only start writ in g paper when you are 90 percent confident with your results." "And you should ask me before."
  "Rmemebr we have many months to do research first."

### Rigour
- 2026-10-17: "For claims and results you should always do hypothesis testing to see how likely it is that it's just luck, and try
  to make things rigorous." Every result: effect with a 95 % bootstrap interval over jets, a null or control with its interval,
  explicit H0/H1 and a p-value, the competing hypotheses' predictions, Holm/BH correction for scans. A read-out is calibrated
  against a matched real change. "When u shift the axis, do you also get the same mass shift YOU would expect?"
- Mechanism claims must be causal: input edits or unit interventions on the real network, with a control.

### Reporting and presentation
- 2026-10-16: "report interesting results to me as plots here, well organized"; "Plots or tables as pngs. Do this whenever there's
  new interesting results." Send them with SendUserFile.
- 2026-10-16: "use a well organized html, clean succinct nonmannered prose no editorial or declarative writing that i can check.
  Each paper should have its own tab, shoudl evolve as you do research. If papers combine the tabs should change as well."
  Tabs = Overview + one per paper; explanations live inside their paper's tab.
- 2026-10-16: "For each claim that you verify you shoudl have a drop down that goes in depth with plots and tables and technical
  description of whatvit is and what you did and the result"; "Also for claims that are in progress."
- 2026-10-17: "Should explain and motivate everything", including the physics and its significance, with images. "Should write
  math in latex form or markdown on documents."
- 2026-10-20: "Make sure everything is described very well also." Every entry says concretely what was edited, what was held
  fixed and why, what the control is, what each number means in words; every symbol, acronym and metric is defined at first use and
  added to the site glossary; units are named by what they compute; one or two sentences on the physics meaning.
- 2026-10-17: "For each result you should at least write a conclusion description of what the result is in plain terms." Label it
  just "Conclusion" (user: not "conclusion in plain terms"). 2–4 sentences: what was found, what it means, what is uncertain;
  at most two numbers; unit IDs only with what they compute. Stored in `site/conclusions.py`.
- 2026-10-16, on prose: "Mannered prose substitutes metaphor and flourish for direct statement. … The fix is to say what you mean.
  When a literal phrase is available, use it. And no editorial or declarative language."
- 2026-10-16: "Notify me with questions as well" (PushNotification; questions also go in `QUESTIONS.md`).

### Compute, files, git
- 2026-10-16: "You should run everything on the cluster. My computer is running something else." No local computation beyond
  seconds-long checks, page builds and small matplotlib renders. "https://nrp.ai this is the cluster rmemebr to follow these rules"
  (see `CLUSTER.md`).
- 2026-10-18: "Pause runs until Monday 9 am, especially running things on the computer to reduce load." Keep local load low.
- 2026-10-16: "Don't touch other people's files as well." On /j-jepa-vol write only under /j-jepa-vol/aaron-interp; every other
  directory (including phat-jet-aaron and other users' runs) is read-only: no writes, no running or importing code from inside it,
  no git. Locally write only in ~/Documents/aaron-interp. Leave other sessions' and people's jobs and pods alone.
- 2026-10-16: "Only use my GitHub." Remote https://github.com/aaronw5/aaron-interp, public ("Public"). "Rmemebr to continuously push
  commit work to git": push after every commit. Never force-push. Commit code and ledger files only, never data, models,
  results, logs, npy/npz/pt.
- 2026-10-16/17: "Continue, and auto resume when usage limit resets"; "Automatically resume on next out of usage as well, until end
  of project" (see `RESUME.md`).
- 2026-10-17: spawn agents as Opus 5.5, effort medium for routine work, high for verification, review and claims.

## 3. Papers (details and current wording in `PAPERS.md`)

| paper | subject | main models | status |
|---|---|---|---|
| A | mechanistic interpretability: statements → objects → variables over layers; the axis-moment mass | JEDI-linear 3-feature n64; other official JEDI checkpoints; small hls4ml models | thesis v2 in force; v3/v3.1 (re-centre on the mass/axis result) proposed, user decision pending, no freeze |
| B | method: the whole network as if-statements (nested MARS), faithfulness, baselines | JEDI-linear | B2 calls 0.85, B2m causal variables 0.4, B4 compactness 0.25 |
| C | circuits in big taggers | ParT_full/kin/kinpid, ParticleNet, L-GATr (to train) | C1 0.8, C2 0.7, C3 0.85, C4 not started |
| D | which physical quantities taggers use, across models and datasets, vs classical taggers | all; hls4ml, JetClass, Top Tagging, Quark–Gluon | D1 0.8, D2 0.7, D4 0.8, D4m 0.45, D3 not started |

Strongest result so far (reviewers, rounds 2–4): JEDI-linear and ParT read the jet mass as the momentum-weighted spread about the
given axis ($\sqrt{M_0 M_2}$, not recentred), not the invariant mass; consequence: dependence on the jet-axis convention.
Reviewer scores (A/B, rounds 2–4): overall 5, 4, 5. Main criticisms: novelty vs EFN/PFN and Deep Sets, single model,
quantisation specifics, off-manifold edits, causal-abstraction standard, degenerate controls, thesis drift.

## 4. The three science loops (AI-Scientist-v2 structure: plan → run → verify → ledger → review per round)

| loop | papers | node ids | reports | running file | round tables | job-name prefix |
|---|---|---|---|---|---|---|
| A/B | A, B (JEDI-linear, small hls4ml models) | E, X, B, R | `inbox/` | `running/A.md` | `results/jedi/round<r>_table.png` | none |
| C | C (ParT, ParticleNet, L-GATr internals) | P (P3 onward) | `inbox_c/` | `running/C.md` | `results/part/c_round<r>_table.png` | `c-` |
| D | D (physics across models/datasets) | H (H1 onward) | `inbox_d/` | `running/D.md` | `results/physics/d_round<r>_table.png` | `d-` |

- Workflow scripts (Claude session directory):
  `~/.claude/projects/-Users-anrunw-Documents-distill-part/97edd5f4-0297-498c-b0c4-c84a470a6b9f/workflows/scripts/`
  `science-loop-jedi-cluster-wf_af611821-cfb.js` (A/B; `-r5.js` = pre-edit copy for resuming round 5),
  `science-loop-bigmodels-cluster-wf_c2606c6d-69e.js` (C), `science-loop-physics-cluster-wf_68571653-bc2.js` (D).
  Args: `{rounds, nodes_per_round, first_round, seed_review}`; seed_review = the previous round's review weaknesses.
- Each planner reads the other loops' `running/*.md` and `TREE.md` to avoid duplicates and writes its own file.
- Only the ledger agent of each loop commits. All loops share one working tree: edit shared files (PAPERS.md, EXPERIMENTS.md,
  TREE.md, site/*.py) with exact Edit replacements of your own sections only; stage only your own hunks (`git add -p`), check
  `git diff --cached`, then commit the staged index with a plain `git commit -m '<msg>'`. Do not use `git commit -- <paths>`: it commits
  the whole working-tree file, including another loop's uncommitted edits to shared files (caught in loop C r6). A plain commit also takes
  whatever another loop staged (happened once, 5780469), so check the staged list first;
  if `.git/index.lock` exists, wait and retry.
- Run ids and resume state: `RESUME.md`.

## 5. Infrastructure

- Repo: `~/Documents/aaron-interp` (git), mirrored to `/j-jepa-vol/aaron-interp/code` by `./sync.sh`.
- Cluster: NRP Nautilus, namespace `cms-ml`, PVC `j-jepa-vol`, image `gitlab-registry.nrp-nautilus.io/jmduarte/weaver:latest`,
  helper pod `anrunw-helper-6b8d8cfdd-x2c94` (file access, tails, tar copies). Labels `owner=anrunw, study=aaron-interp`.
- Job scripts: `./job.sh <name> [cpu mem] '<cmd>'` (CPU; default 1 CPU / 2 Gi, exempt from utilisation rules; `AI_PY=conda` for
  torch); `./part_job.sh <name> '<cmd>'` (GPU, 4 CPU / 16 Gi; venv `/j-jepa-vol/aaron-interp/env-part`, weaver-core 0.4.17);
  `train/gpu_job.sh`, `train/make_jobs.py` (training; see `train/README.md`, `train/STATUS.md`). Logs:
  `/j-jepa-vol/aaron-interp/logs/<name>.log`. Results: `$AI_RESULTS=/j-jepa-vol/aaron-interp/results`, copied back to `results/`.
- energyflow venv: `/j-jepa-vol/aaron-interp/venv-ef`.
- Local matplotlib for tables/figures (seconds): `/Users/anrunw/Documents/jedi-distill/.venv/bin/python`.

### Data and models (volume)
- hls4ml: `/j-jepa-vol/aaron-interp/data` (n64, `_jets`, splits, norm stats incl. `norm_stats_axhard.json`); jets via
  `interp.jedi_core.sample(per_class, seed)`.
- JetClass test files: read-only `/j-jepa-vol/JetClass/Pythia/test_20M/`, listed in `cluster/data_part/test/files.json`; jets via
  `interp.part_core.sample`.
- Top Tagging: read-only `/j-jepa-vol/TopTagging`, `/j-jepa-vol/ParT_top/TopLandscape/{train,val,test}_file.parquet`.
- Quark–Gluon: read-only `/j-jepa-vol/QuarkGluon/{train,test}` (extract only into `/j-jepa-vol/aaron-interp/data/qg`).
- Models: `/j-jepa-vol/aaron-interp/models/{official_models (JEDI-linear), part (ParT_full/kin/kinpid), particlenet
  (ParticleNet_full/kin/kinpid, official weights), small (hls4ml transformers, Linformers, Deep Sets; incl. -axhard, -e100, -p30)}`.
- Earlier ParT/JEDI findings (G-, J-series): `/Users/anrunw/Documents/distill-part/jld-features/research/FINDINGS.md` (read-only).

### Code map
- `interp/jedi_core.py`: exact JEDI-linear forward pass (stages a, P, g, s, u, S, h4, h5, z, logits), overrides, untrained flag.
  Do not change its defaults.
- `interp/part_core.py`: ParT helpers (sample, kin, build, jmass, sub, cat, moments, Attention hooks).
- `interp/<id>_*.py`: one module per experiment; run as `python -m interp.<module>`.
- `jetdistill/`: copied package (ParT network/data; `JETDISTILL_TAGGER` switch; `mars.py`).
- `site/`: results page. `content.py` (overview, papers, claims with detail), `explain.py` (background, glossary, per-claim
  plain/background/motivation, key results, step-by-step explainers registered in `PAPER_EXPLAINERS`), `conclusions.py`
  (CLAIM, EXPLAINER, KEY_RESULT), `tables.py` and `explain_figs*.py` (PNGs), `gen.py` (renders).

### Results page
- Build: `python3 site/gen.py <scratchpad>/site/results_page.html` (copies images to `img/`, writes `files.json`).
- Publish: Artifact tool, `file_path` = that HTML, `files` = new entries of `files.json`, url
  https://claude.ai/artifact/JhMsJ4hbzPfVkCgtj1F6gR ("Tagger Interp Results", private to the user).

## 6. Known pitfalls (each cost time before)
- ParT build: η = jet_eta + sign(jet_eta)·deta. The earlier G192 build mirrored η for jet_eta < 0.
- JetClass rows are ordered by class: never take the first k rows; use random indices or `[::5]`.
- JEDI-linear pooling: the qs zero-bit mask must be applied (`QS = np.asarray(net.L['qs'].iq.quantizer.b)[0] > 0`). Zero-bit
  (slot × channel) entries are pT-rank-window masks.
- Padded particle slots must not count as particles in sums or edits.
- Untrained-network controls are degenerate (one class for every jet). Use shuffled-label, weight-permuted or early-epoch models.
- A rigid translation is off-manifold. Pair it with natural-tail jets (E23) and Δ = 0 controls (X7v).
- `sync.sh` uses two tar streams (`--overwrite` for files, `--skip-old-files` for `.git/objects`).
- Workflow resume: editing a script changes cache keys. Keep a pre-edit copy to resume a running round.
- macOS zsh: avoid `timeout` and variable-as-command patterns; avoid complex `bash -c`.
- In explain.py, LaTeX backslashes need valid Python escapes (raw strings or doubled backslashes).

## 7. Corrections on record (earlier claims changed by later tests)
"√(M0·M2) exact" → the calibrated ratio depends on the size of the change; "10 radial objects" → 5; "W rejection 0.70×" → 1.13×,
not reproduced; "z10 top gate one-sided" → two-sided; "6 pT ramps" → 4; E15 product location withdrawn; C3 v1 "ParT computes the
invariant mass" refuted; A12 scoped to the X4 axis-jump edit; pure first-layer cuts explained by HGQ sparsity (X6v); rank windows
causal only at the hard-particle edge (X6); program not more compact than standard surrogates (B4); no ≤ 50-term program meets
the shift/re-axis criterion (B5).

## 8. Open questions for the user (`QUESTIONS.md`)
- Re-centre Paper A on the mass/axis result (thesis v3/v3.1)? No decision needed yet (exploration phase).
- Deployed L1 input axis definition for JEDI-linear (seed particle, seeded-cone centre, or E-scheme).
- Main JEDI model: keep 3-feature n64 or switch to 16-feature-perminv n64 (82.36 %).
- L-GATr training scale (default in use: the authors' jctagging config on the JetClass 2M-train subset).
- Small-transformer set (open since 2026-10-16; the proposed set has been trained).

## 9. Timeline (working plan, 2026-10-17)
Explore until about mid-December. Then consolidate with the user: choose theses, run confirmatory passes on fresh seeds and
samples. Writing starts only after the user's go-ahead.

## 10. How to brief a new agent
Point it to this file and to the claim or node ids it works on. State which loop it belongs to (node prefix, inbox directory,
job-name prefix), the compute rule (cluster jobs only), the read-only rule, the statistics rule, and that every report starts
with a Conclusion and avoids mannered or editorial prose.
