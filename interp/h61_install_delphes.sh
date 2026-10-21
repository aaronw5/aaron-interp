#!/bin/bash
# H61: install Delphes 3.5.1 (conda-forge) into the pod-local prefix /tmp/env-delphes with micromamba (a CephFS prefix linked at
# ~4 packages/min, d-h61-inst 2026-10-21). Usage inside a job: bash interp/h61_install_delphes.sh && <python command>
set -e
[ -x /tmp/env-delphes/bin/DelphesHepMC2 ] && exit 0
cd /tmp
python3 -c "import urllib.request,tarfile,io; d=urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest',timeout=120).read(); tarfile.open(fileobj=io.BytesIO(d),mode='r:bz2').extract('bin/micromamba','/tmp')"
t0=$(date +%s)
MAMBA_ROOT_PREFIX=/tmp/mamba /tmp/bin/micromamba create -y -q -p /tmp/env-delphes -c conda-forge delphes=3.5.1 > /tmp/mamba.log 2>&1 || { tail -30 /tmp/mamba.log; exit 1; }
rm -rf /tmp/mamba/pkgs
echo "delphes installed in $(( $(date +%s) - t0 ))s: $(ls /tmp/env-delphes/bin | grep -i delphes | tr '\n' ' ')"
