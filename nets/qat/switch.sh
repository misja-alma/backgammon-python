#!/bin/bash
# Once the 32M segment's training is done, restart the runner with the decision plateau rule.
cd /Users/misja/PycharmProjects/backgammon-python
export PYGAME_HIDE_SUPPORT_PROMPT=1
until grep -q "average of the last" nets/qat/f32-576-192-32M.log 2>/dev/null; do sleep 20; done
sleep 5
pkill -f "nets/qat/run.py"; sleep 1; pkill -f "benchmarks.strength"; sleep 3; pkill -f bgsage_server.py
echo "--- f32-576-192: from 32M on the plateau is decided by the decision benchmark (loss, margin 0.005); games every 12M and for the final net" >> nets/qat/results.txt
.venv/bin/python nets/qat/run.py f32-576-192 >> nets/qat/run.out 2>&1
