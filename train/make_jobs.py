"""Write the k8s Job YAMLs for the small-tagger training sets to train/jobs/.

    python train/make_jobs.py [--set main|extra1|e18] [--only n32-s0,n64-s1] [--suffix=-b] [--conc C] [--cpu C] [--mem G]
    kubectl apply -f train/jobs/<job>.yaml

Sets (job name -> runs as model:N:seed[:dir-tag[:train_small flags]], see gpu_job.sh):
  main    7 models x N in {32, 64} x seeds {0, 1} = 28 runs, 4 jobs by (N, seed), 7 runs concurrent per GPU.
  extra1  n64-p30: tf-2b-d16 and tf-2b-d32 at N = 64, seed 1, --patience 30 (reruns of the two early-stopped main runs;
          directory tag -p30, the originals stay); n64-e100: tf-2b-d32 and tf-2b-d64 at N = 64, seeds 0 and 1, 100 epochs
          cosine, --patience 100 (no early stop), directory tag -e100.
  e18     n64-axhard: dsets-d64 and linf-2b-d32-k8 at N = 64, seeds 0 and 1, --axis hardest ((deta, dphi) re-measured
          about the hardest particle; normalisation norm_stats_axhard.json from train/axis_stats.py), tag -axhard.
Several runs share one GPU (NRP: GPU utilisation must stay > 40 %). CPU / memory per set from measurements: about 0.75-1
core and 1.8-2.1 GiB of container memory per concurrent run (main jobs: 5.0-5.1 cores, 12.7-15.0 GiB for 7 runs).
Job names anrunw-ai-train-<job>; labels owner=anrunw, study=aaron-interp, set=small-taggers. The container runs
train/gpu_job.sh from the repo mirror /j-jepa-vol/aaron-interp/code (run ./sync.sh first); the log goes to
/j-jepa-vol/aaron-interp/logs/train-<job>.log.
"""
import argparse
import os

MODELS = ["tf-1b-d16", "tf-2b-d16", "tf-1b-d32", "tf-2b-d32", "linf-1b-d32-k4", "linf-2b-d32-k8", "dsets-d64"]
NS = [32, 64]
SEEDS = [0, 1]
ROOT = "/j-jepa-vol/aaron-interp"

# hostname list: nodes where CephFS metadata ops stalled or GPUs were invisible to CUDA (from the phatjet job generator
# and job.sh); GPU products: those the phatjet runs used (no Volta, no A100/H100: no access on NRP)
BAD_NODES = ("k8s-haosu-24.sdsc.optiputer.net, ry-gpu-01.sdsc.optiputer.net, nautilus-it-gpu05.fullerton.edu, "
             "fiona-prg1.cesnet.cz, k8s-chase-ci-07.calit2.optiputer.net, dtn-gpu2.kreonet.net, "
             "gpn-fiona-mizzou-2.rnet.missouri.edu, rci-nrp-gpu-05.sdsu.edu, node-2-2.sdsc.optiputer.net, "
             "hcc-nrp-shor-c6017.unl.edu, hcc-nrp-shor-c5925.unl.edu, epic001.clemson.edu, "
             "k8s-chase-ci-10.calit2.optiputer.net, nautilus-ext-gpu01.fullerton.edu, chi-dgx-node04.csuchico.edu, "
             "gpu-12.nrp.mghpcc.org, gpu-14.nrp.mghpcc.org, hcc-chase-shor-c4715.unl.edu, k8s-3090-01.usd.edu, "
             "k8s-haosu-22.sdsc.optiputer.net, k8s-haosu-18.sdsc.optiputer.net")
GPUS = ("NVIDIA-GeForce-RTX-3090, NVIDIA-GeForce-RTX-4090, NVIDIA-A10, NVIDIA-A40, NVIDIA-L40, NVIDIA-L40S, NVIDIA-L4, "
        "NVIDIA-RTX-A6000, NVIDIA-TITAN-RTX, NVIDIA-RTX-A5000, NVIDIA-RTX-A4000, Quadro-RTX-6000, "
        "NVIDIA-GeForce-RTX-2080-Ti")

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: anrunw-ai-train-{jname}
  namespace: cms-ml
  labels: {{owner: anrunw, study: aaron-interp, set: small-taggers}}
spec:
  # 1 retry: anrunw-ai-train-n32-s1 (2026-10-16) failed 32 min after start with its pod gone before the container ran
  # (no log, no snapshot) and backoffLimit 0 ended the Job. A retry is safe: gpu_job.sh skips runs that have test.json.
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {{owner: anrunw, study: aaron-interp, set: small-taggers}}
    spec:
      restartPolicy: Never
      activeDeadlineSeconds: 86400
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values: [{bad_nodes}]
              - key: nvidia.com/gpu.product
                operator: In
                values: [{gpus}]
      containers:
      - name: train
        image: gitlab-registry.nrp-nautilus.io/jmduarte/weaver:latest
        command: ["/bin/bash", "-c"]
        args:
        - |
          mkdir -p {root}/logs
          bash {root}/code/train/{script} {job} {conc} {runs} > {root}/logs/train-{jname}.log 2>&1
        resources:
          limits: {{nvidia.com/gpu: "1", memory: "{mem}Gi", cpu: "{cpu}"}}
          requests: {{nvidia.com/gpu: "1", memory: "{mem}Gi", cpu: "{cpu}"}}
        volumeMounts:
        - {{name: vol, mountPath: /j-jepa-vol}}
      volumes:
      - name: vol
        persistentVolumeClaim: {{claimName: j-jepa-vol}}
"""


def job_sets():
    """{set: {job_name: dict(runs=[spec, ...], conc, cpu, mem)}}."""
    main = {f"n{n}-s{s}": dict(runs=[f"{m}:{n}:{s}" for m in MODELS], conc=7, cpu=6, mem=20) for n in NS for s in SEEDS}
    extra1 = {
        "n64-p30": dict(runs=[f"{m}:64:1:-p30:--patience=30" for m in ("tf-2b-d16", "tf-2b-d32")], conc=2, cpu=2, mem=6),
        "n64-e100": dict(runs=[f"{m}:64:{s}:-e100:--epochs=100,--patience=100" for m in ("tf-2b-d32", "tf-2b-d64")
                               for s in SEEDS], conc=4, cpu=4, mem=10),
    }
    e18 = {  # E18: inputs re-measured about the hardest particle (train_small --axis hardest), directory tag -axhard
        "n64-axhard": dict(runs=[f"{m}:64:{s}:-axhard:--axis=hardest" for m in ("dsets-d64", "linf-2b-d32-k8")
                                 for s in SEEDS], conc=4, cpu=4, mem=10),
    }
    # X5: dsets-d64 N = 64: float seed 2; L1 on the phi1 weights (--l1_phi1, directory tag -l1_<lambda>) at 7 lambdas x
    # seeds 0-2 (lambda is chosen on seed 0 only; seeds 1, 2 are trained at every lambda so that no second wave is needed)
    lams = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
    x5runs = ["dsets-d64:64:2"] + [f"dsets-d64:64:{s}:-l1_{l}:--l1_phi1={l}" for s in (0, 1, 2) for l in lams]
    x5 = {"x5-a": dict(runs=x5runs[:8], conc=8, cpu=6, mem=16),
          "x5-b": dict(runs=x5runs[8:15], conc=7, cpu=6, mem=15),
          "x5-c": dict(runs=x5runs[15:], conc=7, cpu=6, mem=15),
          # added after seed 0 reached at most 0.79 zero fraction at lambda 1e-2 (target 0.896, JEDI-linear l1)
          "x5-d": dict(runs=[f"dsets-d64:64:{s}:-l1_{l}:--l1_phi1={l}" for s in (0, 1, 2) for l in ("3e-2", "1e-1")],
                       conc=6, cpu=5, mem=13)}
    # X8 functional nulls: dsets-d64 and jedis-d64 trained on permuted training labels (--shuffle_labels, tag -shuf, 30 epochs
    # without early stopping), and jedis-d64 on true labels (matched trained comparator), seeds 0 and 1
    x8 = {"n64-x8": dict(runs=[f"dsets-d64:64:{s}:-shuf:--shuffle_labels,--patience=100" for s in SEEDS]
                             + [f"jedis-d64:64:{s}:-shuf:--shuffle_labels,--patience=100" for s in SEEDS]
                             + [f"jedis-d64:64:{s}" for s in SEEDS], conc=4, cpu=4, mem=10)}
    # X12: JEDI-linear n64 retrained (train/train_hgq_jedi.py via train/gpu_job_x12.sh, keras/jax/hgq2 env on the GPU); run spec
    # arm:seed[:tag[:flags]] -> models/small/jedi-n64-x12-<arm><tag>/seed<seed>; HGQ seeds 0-2 and the pipeline-matched float arm
    # (quantizers = identity) seeds 0, 1; x12-smoke: 3 epochs on 60,000 jets per arm to measure the epoch time first
    x12 = {"x12-smoke": dict(runs=["hgq:0:-smoke:--epochs=3,--max_train=60000,--max_test=20000,--keep_epochs=2,--save_every=2",
                                   "float:0:-smoke:--epochs=3,--max_train=60000,--max_test=20000,--keep_epochs=2,--save_every=2"],
                             conc=2, cpu=2, mem=8, script="gpu_job_x12.sh"),
           "x12": dict(runs=[f"hgq:{s}" for s in (0, 1, 2)] + [f"float:{s}" for s in (0, 1)], conc=5, cpu=5, mem=16,
                       script="gpu_job_x12.sh")}
    # X9w: input representation x training (2x2): --recentre (angles about the pT-weighted centroid) and --aug 0.08 (random rigid
    # translation of the training jets); Deep Sets d64 N = 64 seeds 0, 1 (tags -rc, -aug, -rcaug) and JEDI-linear HGQ-50 seed 0
    # (jedi-n64-x12-hgq<tag>); the stored / no-aug cell reuses the existing runs. Both sets share one GPU (gpu_job_x9w.sh).
    cells = [("-rc", "--recentre"), ("-aug", "--aug=0.08"), ("-rcaug", "--recentre,--aug=0.08")]
    x9w = {"x9w": dict(runs=[f"dsets-d64:64:{s}:{t}:{f}" for s in SEEDS for t, f in cells] + [f"hgq:0:{t}:{f}" for t, f in cells],
                       conc=6, cpu=6, mem=24, script="gpu_job_x9w.sh")}
    # X9x: augmentation-range scan: Deep Sets d64 N = 64 trained with --aug A for A in {0.04, 0.16, 0.30} rad (stored axis, no
    # recentring), seeds 0, 1, directory tags -aug04 / -aug16 / -aug30 (A = 0 and A = 0.08 reuse the main and X9w -aug runs).
    # PFN / EFN of the same scan run as CPU Jobs (energyflow venv): train/x9w_cpu_batch.sh via job.sh (see interp/x9x_augrange.py).
    x9x = {"x9x": dict(runs=[f"dsets-d64:64:{s}:-aug{t}:--aug={a}" for s in SEEDS for t, a in (("04", "0.04"), ("16", "0.16"), ("30", "0.30"))],
                       conc=6, cpu=6, mem=14)}
    return {"main": main, "extra1": extra1, "e18": e18, "x5": x5, "x8": x8, "x12": x12, "x9w": x9w, "x9x": x9x}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="main", help="main | extra1 | e18 | x5 | x8 | x12 | x9w | x9x")
    ap.add_argument("--only", default="", help="comma-separated job names")
    ap.add_argument("--suffix", default="", help="job-name suffix for resubmits, written --suffix=-b")
    ap.add_argument("--conc", type=int, default=None, help="trainings running at once per GPU (default: the set's)")
    ap.add_argument("--cpu", type=int, default=None, help="default: the set's")
    ap.add_argument("--mem", type=int, default=None, help="Gi (default: the set's)")
    ap.add_argument("--runs", default="", help="override the set's run specs (space-separated), e.g. to change --epochs")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "jobs"), exist_ok=True)
    only = [j for j in args.only.split(",") if j]
    for job, cfg in job_sets()[args.set].items():
        if only and job not in only:
            continue
        conc = args.conc or cfg["conc"]
        cpu = args.cpu or cfg["cpu"]
        mem = args.mem or cfg["mem"]
        jname = job + args.suffix
        path = os.path.join(here, "jobs", f"{jname}.yaml")
        with open(path, "w") as f:
            runs = args.runs.split() if args.runs else cfg["runs"]
            f.write(TEMPLATE.format(jname=jname, job=job, conc=conc, runs=" ".join(runs), root=ROOT,
                                    cpu=cpu, mem=mem, bad_nodes=BAD_NODES, gpus=GPUS, script=cfg.get("script", "gpu_job.sh")))
        print(f"{path}: {len(cfg['runs'])} runs, {conc} concurrent, {cpu} CPU, {mem} Gi")

if __name__ == "__main__":
    main()
