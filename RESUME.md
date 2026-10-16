# Resume checklist (general; first used 2026-10-16 when the API session limit paused all agents)

Rule (user): whenever agents or workflows stop on the API usage limit, resume them automatically after the reset, until the end of the
project. A session cron job fires twice an hour and runs this checklist when it finds rate-limit terminations.

Cluster jobs keep running while agents are paused. On resume:
1. `kubectl -n cms-ml get jobs -l study=aaron-interp`; read finished logs under /j-jepa-vol/aaron-interp/logs/ (e5v, e11-*, e1e3-3f-n64*, p1, train-*).
2. Resume the science loop: Workflow({scriptPath: ".../workflows/scripts/science-loop-jedi-cluster-wf_af611821-cfb.js", resumeFromRunId: "wf_af611821-cfb",
   args: {"rounds":2,"nodes_per_round":3,"first_round":1}}) — round-1 plan (E8, E14, E15) is cached; the run/verify/ledger/review agents rerun.
3. Resume the paused agents by SendMessage ("continue; report where you left off"): E11 MARS fit (a3b876542e12d9f1c), small-transformer
   training (afedb3def8d8fdf99), cluster setup/sweep (a68f05a2790909352), ParT P1 (a9b97a5b584f94169: job anrunw-ai-p1; then p1_plot on the helper pod).
4. E5v: job anrunw-ai-e5v launched by the main session (user OK'd); when done, copy results/jedi/e5_verify.json back and file inbox/E5v_verify.md.
5. Merge inbox → ledger (EXPERIMENTS/TREE/PAPERS/site/content.py), regenerate the page (python3 site/gen.py <scratchpad>/results_page.html), publish
   the artifact, send new PNGs to the user, commit + push.
