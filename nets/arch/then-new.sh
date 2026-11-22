#!/bin/bash
# Waits for the running sweep to finish, then restarts it: the state keeps the finished nets,
# and 512-256 and 756 (added to NETS) start from scratch.
cd /Users/misja/PycharmProjects/backgammon-python
while pgrep -f "nets/arch/run.py" >/dev/null; do sleep 60; done
.venv/bin/python nets/arch/run.py >> nets/arch/run.out 2>&1
