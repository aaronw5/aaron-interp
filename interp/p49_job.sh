#!/bin/bash
# P49: ./part_job.sh with k8s-chase-ci-01 added to the excluded nodes (device plugin there reports "GPU is lost";
# UnexpectedAdmissionError on 2026-10-21, as for P44). Same usage: interp/p49_job.sh <name> '<command>'
set -euo pipefail
D=$(cd "$(dirname "$0")/.." && pwd)
sed 's/values: \[k8s-haosu-24.sdsc.optiputer.net,/values: [k8s-chase-ci-01.calit2.optiputer.net, k8s-haosu-21.sdsc.optiputer.net, k8s-haosu-24.sdsc.optiputer.net,/' "$D/part_job.sh" > /tmp/p49_part_job.sh
grep -q 'k8s-chase-ci-01' /tmp/p49_part_job.sh || { echo 'node exclusion not inserted' >&2; exit 1; }
bash /tmp/p49_part_job.sh "$@"
