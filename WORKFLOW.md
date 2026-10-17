# Research loop (AI-Scientist-v2 structure, run by Claude + subagents)

This loop follows the structure of SakanaAI/AI-Scientist-v2: ideas → staged agentic tree search over experiments → figure
review → write-up → automated review → new nodes. The LLM is Claude in this session. Workers are subagents launched with
the Agent tool; the cluster takes the large sweeps. The user asked for no separate sandbox or API-key setup.

## 1. Ideas (`ideas/*.json`)
Fields follow v2: Name, Title, Short Hypothesis, Related Work, Abstract, Experiments, Risk Factors and Limitations.
- One file per paper (A–D).
- The working thesis and its version history live in `PAPERS.md`.
- A novelty check against the literature runs before each paper's write-up starts.

## Phase (user, 2026-10-17): exploration
"Don't freeze paper right, we should explore more." "What I mean is we have 3 months until deadline we can explore." Deadline ≈ mid-January 2027
(exact date to confirm). Working plan: explore until ≈ mid-December (≥ 2 of 3 loop nodes per round exploratory); then consolidate
(choose theses with the user, confirmatory runs on fresh seeds and samples); writing only after the user's go-ahead (write-up gate). Theses are working hypotheses; no freezing, no confirmatory-only rounds yet.
Each round mixes rigour on existing claims with exploration: new phenomena, other models (ParticleNet, LorentzNet, L-GATr,
PELICAN-type invariant models), other datasets (top tagging, quark–gluon), other physics quantities (beyond mass: N-subjettiness,
multiplicity, b-tagging, leptons), and other methods (DAS, SAEs, EFP bases). Statistics rules still apply to every result.

## Loops per paper (user, 2026-10-16)
One science loop per paper. When the evidence diverges from a paper's thesis, or a result points to a paper that does not
exist yet, a new thesis goes into PAPERS.md and a new loop starts on it. At most 3 loops run at once. Papers that become too similar, or that cannot stand alone, are combined (criteria in PAPERS.md).

## 2. Experiment tree (`TREE.md`)
Each node records:
- id
- parent
- stage
- paper/claim
- plan
- script
- status: draft / running / buggy / done / pruned
- metric(s)
- figure
- review note

The four stages, as in v2:
1. **Initial implementation:** the measurement works and is verified against an exact check (identities, reproduction of the network's accuracy).
2. **Baselines and controls:** untrained network, random-unit or random-direction controls, raw-input readouts.
3. **Research agenda:** the claims in PAPERS.md, one node per claim, several drafts in parallel when the design is uncertain.
4. **Ablations and causal tests:** stage substitution, input edits, interventions on units, robustness across the 20 JEDI models.

Rules:
- **Best-first:** expand next the node whose result would most change a thesis claim, or the paper narrative.
- **Workers:** up to 3–4 workers at once.
- **Debugging:** a buggy node gets at most 2 debug attempts, then it is pruned with the reason recorded.

## 3. Figure review
Every node that produces a figure has it opened and checked: readable at a glance, axes in physical units, one message.
The figure is redone if it fails.

## 4. Write-up (`paper/`)
**Gate (user, 2026-10-16):** no paper writing starts until the results behind that paper reach about 90 % confidence, and the
user is asked first. The timeline is several months of research; depth comes before writing. Until the gate is passed, this
stage is limited to keeping PAPERS.md (theses, claims, confidence per claim) current.

- A LaTeX draft per paper in ICML format.
- Sections are filled as claims reach stage 4.
- Citations come from IDEAS.md's literature list, plus searches for each claim.

## 5. Automated review
- A reviewer subagent scores each draft on the ICML review form: soundness, presentation, contribution, overall, confidence.
- It lists the weaknesses and the experiments that would address them.
- Each weakness becomes a node in TREE.md, at stage 2–4.

## Statistics and rigour (user 2026-10-17: "always do hypothesis testing to see how likely it is that it's just luck")
Every claim and result reports:
- the effect size with a 95 % interval (bootstrap over jets; seeds where models are trained);
- the matching null or control (an edit or intervention that should not produce the effect: random units, rotation, untrained
  network, shuffled labels) with its own interval, and a test of effect vs null (permutation or bootstrap p-value; McNemar for paired
  call changes); state H0 and H1 explicitly;
- when one hypothesis is compared with alternatives (e.g. invariant mass vs √(M0·M2) vs Σ pT r), the prediction of each and which
  are excluded at what significance;
- read-out calibration: a probe fitted on ordinary jets shrinks toward the mean, so any "observed / predicted" ratio from a probe is
  also reported relative to the probe's response to a real change of the same predicted size (matched calibration edit);
- multiple comparisons: when many units or observables are scanned, Holm or Benjamini–Hochberg correction, and the number tested;
- N (jets), seeds, and which jets were used for fitting vs testing.
Verifiers check these and rerun with a different seed.

## Documents (user 2026-10-16)
- Math in LaTeX: `$...$` in Markdown files (GitHub renders it) and on the results page (MathJax); no Unicode pseudo-math in documents.
- Explain and motivate everything: every claim, figure and table on the page has background (what the reader needs to know),
  motivation (why it is tested), the physics meaning and why it matters, and a plain one-line summary (site/explain.py).
- Use explanatory figures (diagrams on real jets) where a result is not obvious from numbers (site/explain_figs.py).
- Every result gets a step-by-step explainer in its paper's tab (question → background → method → result with figure → checks →
  why it matters), like the mass explainer (user: "Everything should be explained like this"). Layer letters are defined in the
  Paper A "How to read this paper" explainer.
- Page build: `python3 site/gen.py <scratchpad>/site/results_page.html` writes the page + img/ + files.json; publish with the
  Artifact `url` of the results page and `files` = files.json (images published once, not inlined).
- Organisation (user 2026-10-17): tabs are Overview + one per paper, nothing else. A result's explanation lives in the tab of the
  paper it belongs to (site/explain.py PAPER_EXPLAINERS), with cross-references from other papers' claims; the Overview holds only
  short summaries that point to the paper tab.

- Conclusion (user, 2026-10-17; label it just "Conclusion": "For each result you should at least write a conclusion description of what the result is in plain terms"): every result — claim drop-down, explainer, key result, round table row, report, chat summary — carries 2–4 plain sentences: what was found, what it means for the question, what is still uncertain. At most two numbers; unit IDs only with what they compute; terms defined. Stored in site/conclusions.py (CLAIM, EXPLAINER, KEY_RESULT) and rendered by site/gen.py at the top of each drop-down.

## Writing rule (all text: ledger, reports, later papers)
No mannered prose: no metaphor or flourish in place of a direct statement ("a dial worth turning" → "a parameter worth
varying"). When a literal phrase exists, use it. No editorial or declarative language; interpretations scoped to the
measurement.

## Questions
Questions for the user go out as a notification (PushNotification) and are listed in QUESTIONS.md with the date and the answer.

## Results page (user 2026-10-16)
One tabbed HTML page, one tab per paper (`site/content.py` → `site/gen.py` → published artifact "Tagger Interp Results",
https://claude.ai/artifact/JhMsJ4hbzPfVkCgtj1F6gR). Plots and tables as PNGs (`site/tables.py`), also sent to the user whenever a
result lands. The ledger step updates `site/content.py` each round: every claim carries a `detail` block (what the claim is, procedure with
sample/edit/control/metric, result numbers, tables, figures, verification, files) rendered as a drop-down (user 2026-10-16); tabs
merge when papers merge. Prose: succinct, no verdict or editorial language.

## Git (user 2026-10-16)
Remote: https://github.com/aaronw5/aaron-interp (public; the user's account only). Every commit is pushed right away
(`git push` after each commit by the ledger step and by me). Code and ledger files only; data, models, results and logs stay out.

## Ledger
- `EXPERIMENTS.md`: one row per node result, with the effect on the thesis: supports / weakens / refines.
- Thesis changes go in `PAPERS.md`, versioned.
