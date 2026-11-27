#!/bin/bash
# Waits for the architecture sweep to finish, rebuilds the Python extension (int8 eval nets need
# the new loader; never while a benchmark runs), then runs the quantization-aware training.
cd /Users/misja/PycharmProjects/backgammon-python
export PYGAME_HIDE_SUPPORT_PROMPT=1
until ! pgrep -f "nets/arch/run.py" >/dev/null; do sleep 60; done
sleep 10
make build >> nets/qat/run.out 2>&1
.venv/bin/python nets/qat/run.py >> nets/qat/run.out 2>&1
