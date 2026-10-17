"""Write the k8s Job YAMLs for the small-tagger training set to train/jobs/.

    python train/make_jobs.py [--only n32-s0,n64-s1] [--suffix=-b] [--conc 7] [--cpu 6] [--mem 16]
    kubectl apply -f train/jobs/<job>.yaml

Set: 7 models x N in {32, 64} x seeds {0, 1} = 28 runs, grouped into 4 jobs by (N, seed), the 7 models of a job training
concurrently on one GPU (NRP: GPU utilisation must stay > 40 %; one tiny model alone does not). Job names
anrunw-ai-train-<job>; labels owner=anrunw, study=aaron-interp. The container runs train/gpu_job.sh from the repo
mirror /j-jepa-vol/aaron-interp/code (run ./sync.sh first); the log goes to /j-jepa-vol/aaron-interp/logs/train-<job>.log.
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
          bash {root}/code/train/gpu_job.sh {job} {conc} {runs} > {root}/logs/train-{jname}.log 2>&1
        resources:
          limits: {{nvidia.com/gpu: "1", memory: "{mem}Gi", cpu: "{cpu}"}}
          requests: {{nvidia.com/gpu: "1", memory: "{mem}Gi", cpu: "{cpu}"}}
        volumeMounts:
        - {{name: vol, mountPath: /j-jepa-vol}}
      volumes:
      - name: vol
        persistentVolumeClaim: {{claimName: j-jepa-vol}}
"""


def jobs():
    """{job_name: [model:N:seed, ...]} — one job per (N, seed), all 7 models."""
    out = {}
    for n in NS:
        for s in SEEDS:
            out[f"n{n}-s{s}"] = [f"{m}:{n}:{s}" for m in MODELS]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated job names")
    ap.add_argument("--suffix", default="", help="job-name suffix for resubmits (e.g. -b)")
    ap.add_argument("--conc", type=int, default=7, help="trainings running at once per GPU")
    ap.add_argument("--cpu", type=int, default=6)
    ap.add_argument("--mem", type=int, default=16, help="Gi")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "jobs"), exist_ok=True)
    only = [j for j in args.only.split(",") if j]
    for job, runs in jobs().items():
        if only and job not in only:
            continue
        jname = job + args.suffix
        path = os.path.join(here, "jobs", f"{jname}.yaml")
        with open(path, "w") as f:
            f.write(TEMPLATE.format(jname=jname, job=job, conc=args.conc, runs=" ".join(runs), root=ROOT,
                                    cpu=args.cpu, mem=args.mem, bad_nodes=BAD_NODES, gpus=GPUS))
        print(f"{path}: {len(runs)} runs, {args.conc} concurrent, {args.cpu} CPU, {args.mem} Gi")


if __name__ == "__main__":
    main()
