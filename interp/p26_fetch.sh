#!/bin/bash
# P26: download the released Sophon weights (jet-universe/sophon on Hugging Face: models/JetClassII_Sophon/model.pt, model.onnx;
# the same files are in github.com/jet-universe/sophon) into /j-jepa-vol/aaron-interp/models/sophon, once. Nothing is trained.
set -euo pipefail
D=${SOPHON_DIR:-/j-jepa-vol/aaron-interp/models/sophon}; mkdir -p "$D"
for f in model.pt model.onnx; do
  [ -s "$D/$f" ] || curl -sSL --retry 3 -o "$D/$f" "https://huggingface.co/jet-universe/sophon/resolve/main/models/JetClassII_Sophon/$f"
done
ls -l "$D"; sha256sum "$D"/model.pt "$D"/model.onnx
