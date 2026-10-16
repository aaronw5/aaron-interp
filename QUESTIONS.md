# Questions for the user

| date | question | status | answer |
|---|---|---|---|
| 2026-10-16 | L-GATr: the only JetClass checkpoints on the volume are other groups' runs (jetclass_2m_benchmark_runs/lgatr, 1p3mFLOPs_runs/lgatr). Copy a checkpoint into aaron-interp and use it read-only, or train our own? | answered 2026-10-16 | Train our own, or use the checkpoint in the authors' GitHub repo. Rule: always check first whether weights for the task are available online. |
| 2026-10-16 | Small hls4ml transformers: proposed set = native transformer (1 and 2 blocks, d = 16/32, 2 heads) and Linformer (k = 4/8), N = 32 and 64, 2 seeds each, ~8 GPU jobs. OK, or a different set? | open | |
| 2026-10-16 | L-GATr weights online: none released (checked heidelberg-hepml/lgatr and lorentz-gatr READMEs/releases, PyPI, Zenodo, HF; tagger-quantization pending). So: train our own. Proposed: authors' `jctagging` config from lorentz-gatr, on the JetClass 2M-train subset the volume's benchmark runs used (not the 100M, 10^6-iteration paper setup), 1 GPU job. OK with this scale? | open | |
| 2026-10-16 | "Best version of JEDI": taken as 3-feature n64 (the model behind all J-results, 80.9 %). If the 16-feature n128 variant is the one you mean, say so and I switch (the 16-feature inputs still need reproducing). | assumed 3-feature n64 | |

