#!/bin/bash
# Mirror this repo to the cluster volume: /j-jepa-vol/aaron-interp/code becomes the same git repository (same .git, same
# working files). Copies .git plus every file git tracks or would track (git ls-files -co --exclude-standard, so ignored
# data/, results/, logs/, __pycache__ stay local), through the helper pod, overwriting in place. It removes nothing:
# files that exist on the cluster but no longer locally are listed at the end (delete them by hand if wanted).
#   ./sync.sh
set -euo pipefail
NS=cms-ml
DEST=/j-jepa-vol/aaron-interp/code
REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)
POD=$(kubectl -n $NS get pods -l owner=anrunw -o name | sed 's|pod/||' | grep '^anrunw-helper-' | head -1)
[ -n "$POD" ] || { echo "no anrunw-helper pod" >&2; exit 1; }
cd "$REPO"
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
git ls-files -co --exclude-standard -z > "$LIST"
kubectl -n $NS exec "$POD" -- mkdir -p "$DEST"
# working files and mutable .git state (refs, index, HEAD, logs) overwrite; .git/objects are content-addressed and
# read-only once written, so existing ones are skipped (overwriting them made tar exit with "Cannot open: File exists")
COPYFILE_DISABLE=1 tar cf - --null -T "$LIST" --exclude=.git/objects .git | kubectl -n $NS exec -i "$POD" -- tar xf - -C "$DEST" --no-same-owner --overwrite
COPYFILE_DISABLE=1 tar cf - .git/objects | kubectl -n $NS exec -i "$POD" -- tar xf - -C "$DEST" --no-same-owner --skip-old-files
echo "synced $(tr -cd '\0' < "$LIST" | wc -c | tr -d ' ') files + .git -> $POD:$DEST"
REMOTE=$(kubectl -n $NS exec "$POD" -- find "$DEST" -type f -not -path "$DEST/.git/*" -not -path '*/__pycache__/*' | sed "s|^$DEST/||")
STALE=$(comm -23 <(printf '%s\n' "$REMOTE" | sort) <(tr '\0' '\n' < "$LIST" | sort) | grep -v '^$' || true)
[ -z "$STALE" ] || printf 'on the cluster but not local (not removed):\n%s\n' "$STALE"
echo "local:   HEAD $(git rev-parse --short HEAD 2>/dev/null || echo none), status $(git status --porcelain | md5)  $(git status --short | wc -l | tr -d ' ') entries"
echo "cluster: HEAD $(kubectl -n $NS exec "$POD" -- git -C "$DEST" rev-parse --short HEAD 2>/dev/null || echo none), status $(kubectl -n $NS exec "$POD" -- git -C "$DEST" status --porcelain | md5)  $(kubectl -n $NS exec "$POD" -- git -C "$DEST" status --short | wc -l | tr -d ' ') entries"
