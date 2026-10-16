"""Download chosen ROOT files out of JetClass's tar archives on zenodo without downloading the whole archive: the tar headers
are read with HTTP range requests (512 bytes each), then only the chosen members are fetched.

  python -m jetdistill.part.fetch list test                         # the files of the test archive (20M jets, 200 files)
  python -m jetdistill.part.fetch get test 2 data_part/raw/test     # the first 2 files of every class (2M jets, 10 %)
  python -m jetdistill.part.fetch get val 1 data_part/raw/val       # 1 file of every class of the validation archive"""
import json, re, sys, tarfile, time
from pathlib import Path
import requests

URL = dict(test='https://zenodo.org/records/6619768/files/JetClass_Pythia_test_20M.tar?download=1',
           val='https://zenodo.org/records/6619768/files/JetClass_Pythia_val_5M.tar?download=1')
INDEX = Path('data_part/raw')


def _range(url, a, b, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=120); r.raise_for_status(); return r.content
        except Exception:
            if k == tries - 1: raise
            time.sleep(2 ** (k + 1))


def index(which):
    """[(name, data offset, size)] of every file in the archive (cached in data_part/raw/<which>_index.json)"""
    f = INDEX / f'{which}_index.json'
    if f.exists(): return json.loads(f.read_text())
    url, off, out = URL[which], 0, []
    while True:
        h = _range(url, off, off + 511)
        if len(h) < 512 or h == b'\0' * 512: break
        ti = tarfile.TarInfo.frombuf(h, 'utf-8', 'surrogateescape')
        if ti.isfile(): out.append((ti.name, off + 512, ti.size))
        off += 512 + (ti.size + 511) // 512 * 512
    INDEX.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out)); return out


def choose(which, per_class):
    """the first per_class files of each class (names like HToBB_100.root)"""
    by = {}
    for name, off, size in index(which):
        m = re.match(r'(.+)_(\d+)\.root$', Path(name).name)
        if m: by.setdefault(m.group(1), []).append((int(m.group(2)), name, off, size))
    return {c: [x[1:] for x in sorted(v)[:per_class]] for c, v in sorted(by.items())}


def get(which, per_class, dest, chunk=64 << 20):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True); url = URL[which]; files = []
    for c, lst in choose(which, per_class).items():
        for name, off, size in lst:
            p = dest / Path(name).name; files.append(str(p))
            if p.exists() and p.stat().st_size == size: continue
            t0 = time.time(); tmp = p.with_suffix('.part')
            with open(tmp, 'wb') as fo:
                for a in range(off, off + size, chunk): fo.write(_range(url, a, min(off + size, a + chunk) - 1))
            assert tmp.stat().st_size == size; tmp.rename(p); print(f'{p.name}: {size / 1e6:.0f} MB in {time.time() - t0:.0f} s', flush=True)
    return files


if __name__ == '__main__':
    cmd, which, *rest = sys.argv[1:]
    if cmd == 'list':
        for name, off, size in index(which): print(name, size)
    else:
        get(which, int(rest[0]), rest[1])
