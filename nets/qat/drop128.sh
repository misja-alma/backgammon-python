#!/bin/bash
# When the runner starts the f32-128-128 segment, stop it and continue with f32-576-192 only.
cd /Users/misja/PycharmProjects/backgammon-python
export PYGAME_HIDE_SUPPORT_PROMPT=1
until [ -e nets/qat/f32-128-128-4M.log ]; do sleep 20; done
pkill -f "nets/qat/run.py"; sleep 1; pkill -f "f32-128-128"; sleep 2
rm -f nets/qat/f32-128-128-4M*
python3 - <<'PY'
import json; p = "nets/qat/state.json"; s = json.load(open(p)); s["runs"].pop("f32-128-128", None); json.dump(s, open(p, "w"), indent=1)
PY
echo "--- f32-128-128 dropped: the existing 128-128 nets are converted instead" >> nets/qat/results.txt
.venv/bin/python nets/qat/run.py f32-576-192 >> nets/qat/run.out 2>&1
