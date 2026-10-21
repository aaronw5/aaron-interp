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
- 2026-10-19: A/B r5 done (commit 949a504); A/B r6 wf_55b49d75-ada (main JEDI script, first_round 6, seed = round-5 review).
- 2026-10-19: A/B r6 resumed (wf_55b49d75-ada), D r2 resumed (wf_53cb1527-b8a) at 14:04; C r1 done (commit d9b4eb0); C r2 wf_a060c9f5-9eb (main bigmodels script, first_round 2, seed = C1 review). L-GATr training anrunw-ai-c-p5-lgatr2 until ~2026-10-21 05:00 UTC.
- 2026-10-19: D r2 done (commit 2e3f127); D r3 wf_441f231d-b06 (main physics script, first_round 3, seed = D2 review).
- 2026-10-19 19:29: A/B r6, C r2, D r3 resumed after the 7 pm reset. A/B r6 done (commit 89b3b13); A/B r7 wf_e675b403-7ee (main JEDI script, first_round 7, seed = round-6 review).
- 2026-10-19: C r2 done (content in 5780469, message commit 63e64c5); C r3 wf_9459e25c-05d (main bigmodels script, first_round 3, seed = C2 review). P10 ParT retrains ETA 2026-10-20 05-06 UTC.
- 2026-10-19: D r3 done (commit 2fd1e98); D r4 wf_9d42d1b6-f6e (main physics script, first_round 4, seed = open questions after D3).
- 2026-10-20 00:29: resumed after the 00:20 reset. A/B r7 with science-loop-jedi-cluster-r7.js (pre-edit copy) wf_e675b403-7ee; C r3 with science-loop-bigmodels-cluster-c3.js (pre-edit copy) wf_9459e25c-05d; D r4 main physics script wf_9d42d1b6-f6e. Next rounds use the main scripts.
- 2026-10-20: C r3 done (commit bfa91a1); C r4 wf_e40d2721-c8f (main bigmodels script, first_round 4).
- 2026-10-20: A/B r7 done (commit f3eb4e8); A/B r8 wf_dd17a094-909 (main JEDI script, first_round 8).
- 2026-10-20: D r4 done (commit bf79a0a); D r5 wf_087a3804-148 (main physics script, first_round 5).
- 2026-10-20 05:29: A/B r8, C r4, D r5 resumed after the 05:20 reset. C r4 done (commit 8b337de); C r5 wf_c597aea9-658 (main bigmodels script, first_round 5).
- 2026-10-20: D r5 done (commit b68644e); D r6 wf_18088c88-26c (main physics script, first_round 6).
- 2026-10-20: A/B r8 done (commit e57897a); A/B r9 wf_b794e1fe-dd0 (main JEDI script, first_round 9).
- 2026-10-20 10:29: C r5, D r6, A/B r9 resumed after the 10:20 reset. D r6 done (commit 3d483bf); D r7 wf_bd8a741e-773 (main physics script, first_round 7). fiona8.ucsc.edu excluded in part_job.sh.
- 2026-10-20: C r5 done (commit 6278b74); C r6 wf_c59ffa74-7d9 (main bigmodels script, first_round 6).
- 2026-10-20: C r6 done (commit f036ff2); C r7 wf_c597aea9... no: C r7 wf_7f21c30f-d84 (main bigmodels script, first_round 7). Ledger commit rule changed (stage own hunks, plain commit); only ledger prompts changed, so resuming A/B r9 and D r7 with the main scripts keeps plan/run/verify cached.
- 2026-10-20: A/B r9 done (commit 8ec72a5); A/B r10 wf_e1f7ef15-c31 (main JEDI script, first_round 10).
- 2026-10-20 15:29: A/B r10, C r7, D r7 resumed after the 15:20 reset. A/B r10 done (commit 390b9f3); A/B r11 wf_1d8d2ad4-fae (main JEDI script, first_round 11).
- 2026-10-20: D r7 done (commit 99017a5); D r8 wf_0c1e4d3c-169 (main physics script, first_round 8).
- 2026-10-20: describe-everything-well rule added to RULES of all three scripts. To resume the rounds running at that time use the pre-edit copies: A/B r11 science-loop-jedi-cluster-r11.js (wf_1d8d2ad4-fae), C r7 science-loop-bigmodels-cluster-c7.js (wf_7f21c30f-d84), D r8 science-loop-physics-cluster-d8.js (wf_0c1e4d3c-169).
- 2026-10-20: C r7 done (commit af4d385); C r8 wf_53ea4ce7-261 (main bigmodels script, first_round 8, with the describe-everything-well rule). k8s-haosu-22 excluded.
- 2026-10-20: D r8 done (commit de85544); D r9 wf_0d7cef81-b5e (main physics script, first_round 9). k8s-haosu-09 excluded.
- 2026-10-20: C r8 done (commit 3df4d46); C r9 wf_498e7b84-c32 (main bigmodels script, first_round 9).
- 2026-10-20: A/B r11 done (commit 36ac2ac); A/B r12 wf_2ebf6ee9-f33 (main JEDI script, first_round 12).
- 2026-10-20: D r9 done (commit 65b08f0); D r10 wf_b1f0e8ff-7bb (main physics script, first_round 10).
- 2026-10-20: A/B r12 done (commit 3a9de7a); A/B r13 wf_a1f14b57-3fe (main JEDI script, first_round 13).
- 2026-10-21: C r9 done (commit e70885b); C r10 wf_d592ce1f-717 (main bigmodels script, first_round 10).
- 2026-10-21: D r10 done (commit f1f40c7); D r11 wf_18b0d6c8-7f1 (main physics script, first_round 11).
- 2026-10-21: A/B r13 done (commit 042b9e3); A/B r14 wf_b516dd4b-d60 (main JEDI script, first_round 14).
- 2026-10-21: D r11 done (commit 63a9bfa); D r12 wf_d0c742b0-841 (main physics script, first_round 12).
- 2026-10-21: A/B r14 done (commit 674379c); A/B r15 wf_8d15945d-f7f (main JEDI script, first_round 15).
- 2026-10-21: C r10 done (commit 37a7407); C r11 wf_1e25fbd2-baa (main bigmodels script, first_round 11).
- 2026-10-21: D r12 done (commit 9960c1f); D r13 wf_2bcd36a3-229 (main physics script, first_round 13).
- 2026-10-21: A/B r15 done (commit e1ec618); A/B r16 wf_261a9c55-ce3 (main JEDI script, first_round 16).
- 2026-10-22: D r13 done (commit 10e2854); D r14 wf_90200f52-3be (main physics script, first_round 14).
- 2026-10-22: A/B r16 done (commit 9e50b17); A/B r17 wf_28756b90-115 (main JEDI script, first_round 17).
- 2026-10-22: C r11 done (commit 8ed1cfe); C r12 wf_d081833a-45d (main bigmodels script, first_round 12; P51 = P46 relaunch first). k8s-chase-ci-01 excluded.
- 2026-10-22: D r14 done (commit c03f5e2); D r15 wf_4e5529a2-85e (main physics script, first_round 15; H66 relaunch first).
- 2026-10-22: A/B r17 done (commit db012d2); A/B r18 wf_36780dad-f18 (main JEDI script, first_round 18). job.sh now tracked.
