# Resume checklist (general; first used 2026-10-16 when the API session limit paused all agents)

Rule (user): whenever agents or workflows stop on the API usage limit, resume them automatically after the reset, until the end of the
project. A session cron job fires twice an hour and runs this checklist when it finds rate-limit terminations.

Cluster jobs keep running while agents are paused. On resume:
1. `kubectl -n cms-ml get jobs -l study=aaron-interp`; read finished logs under /j-jepa-vol/aaron-interp/logs/ (e5v, e11-*, e1e3-3f-n64*, p1, train-*).
2. Resume the science loop from its LAST run id (currently round 5: wf_d4e53fbf-fcc, args {"rounds":1,"nodes_per_round":3,"first_round":5,"seed_review":"<round-4 review: functional nulls, EFN/PFN, X5>"}).
   If the script was edited since the run started, a resume reruns every agent (cache keys include the prompt); in that case merge
   the finished agents' results from journal.jsonl with a single ledger agent instead (as done for round 1 → commit 0511d51).
3. Resume the paused agents by SendMessage ("continue; report where you left off"): E11 MARS fit (a3b876542e12d9f1c), small-transformer
   training (afedb3def8d8fdf99), cluster setup/sweep (a68f05a2790909352), ParT P1 (a9b97a5b584f94169: job anrunw-ai-p1; then p1_plot on the helper pod).
4. E5v: job anrunw-ai-e5v launched by the main session (user OK'd); when done, copy results/jedi/e5_verify.json back and file inbox/E5v_verify.md.
5. Merge inbox → ledger (EXPERIMENTS/TREE/PAPERS/site/content.py), regenerate the page (python3 site/gen.py <scratchpad>/results_page.html), publish
   the artifact, send new PNGs to the user, commit + push.

## Current state (2026-10-19): three loops
| loop | script (in ~/.claude/projects/-Users-anrunw-Documents-distill-part/97edd5f4-0297-498c-b0c4-c84a470a6b9f/workflows/scripts/) | last run id | args |
|---|---|---|---|
| A/B round 5 | science-loop-jedi-cluster-wf_af611821-cfb-r5.js (pre-edit copy; round 6+ uses science-loop-jedi-cluster-wf_af611821-cfb.js) | see the latest launch in this file's log below | {"rounds":1,"nodes_per_round":3,"first_round":5,"seed_review":"<round-4 review: functional nulls, EFN/PFN, X5>"} |
| C round 1 | science-loop-bigmodels-cluster-wf_c2606c6d-69e.js | see log | {"rounds":1,"nodes_per_round":3,"first_round":1,"seed_review":"<Paper C starting review>"} |
| D round 1 | science-loop-physics-cluster-wf_68571653-bc2.js | see log | {"rounds":1,"nodes_per_round":3,"first_round":1,"seed_review":"<Paper D starting review>"} |

Resume = Workflow({scriptPath, resumeFromRunId: <last run id>, args: <same args>}). The exact args of each launch are in the
workflow diagnostics of the session (journal.jsonl under subagents/workflows/<run id>/).

Auto-resume cron prompt (session cron, 23,53 * * * *, 7-day expiry → re-create):
"Auto-resume check (user rule 2026-10-16: "Automatically resume on next out of usage as well, until end of project"). Do: (1) look at the most recent task notifications / agent handbacks in this session for agents or workflows that terminated with an API rate-limit error (HTTP 429 "session limit … resets HH:MM"); (2) if there are any and the stated reset time has passed (check `date` in America/New_York), resume them: re-launch the science-loop Workflow with its scriptPath + resumeFromRunId from the last launch, and SendMessage each dead agent "the limit has reset; continue where you left off and report"; follow aaron-interp/RESUME.md; (3) if the reset time has not passed, do nothing; (4) if nothing died, do nothing and reply with one line. Also: if this is the final firing of this cron job (7-day auto-expiry), re-create it with the same schedule and prompt. Keep the reply to one or two lines unless something was resumed."

## Run-id log
- 2026-10-17: A/B r5 wf_d4e53fbf-fcc; C r1 wf_c2606c6d-69e; D r1 wf_68571653-bc2 (all stopped on the weekly limit; paused by the user until Mon 2026-10-19 09:00).
- 2026-10-19: D r1 done (commit b0dc3bd); D r2 wf_53cb1527-b8a (main physics script, first_round 2, seed = D1 review).
