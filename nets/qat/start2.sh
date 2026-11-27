#!/bin/bash
# Waits for the 256-256 QAT runs to finish, then trains 128-128 in int8 and 576-192 in f16
# from scratch, round-robin, same schedule and benchmark.
cd /Users/misja/PycharmProjects/backgammon-python
export PYGAME_HIDE_SUPPORT_PROMPT=1
until ! pgrep -f "nets/qat/run.py" >/dev/null; do sleep 60; done
sleep 10
.venv/bin/python nets/qat/run.py i8-128-128 f16-576-192 >> nets/qat/run.out 2>&1
