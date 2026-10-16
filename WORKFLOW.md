# Research loop (AI-Scientist-v2 structure, run by Claude + subagents)

This loop follows the structure of SakanaAI/AI-Scientist-v2: ideas → staged agentic tree search over experiments → figure
review → write-up → automated review → new nodes. The LLM is Claude in this session. Workers are subagents launched with
the Agent tool; the cluster takes the large sweeps. The user asked for no separate sandbox or API-key setup.

## 1. Ideas (`ideas/*.json`)
Fields follow v2: Name, Title, Short Hypothesis, Related Work, Abstract, Experiments, Risk Factors and Limitations.
- One file per paper (A–D).
- The working thesis and its version history live in `PAPERS.md`.
- A novelty check against the literature runs before each paper's write-up starts.

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

## Writing rule (all text: ledger, reports, later papers)
No mannered prose: no metaphor or flourish in place of a direct statement ("a dial worth turning" → "a parameter worth
varying"). When a literal phrase exists, use it. No editorial or declarative language; interpretations scoped to the
measurement.

## Questions
Questions for the user go out as a notification (PushNotification) and are listed in QUESTIONS.md with the date and the answer.

## Ledger
- `EXPERIMENTS.md`: one row per node result, with the effect on the thesis: supports / weakens / refines.
- Thesis changes go in `PAPERS.md`, versioned.
