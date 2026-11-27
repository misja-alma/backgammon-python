#!/bin/bash
# Hands the money run over to benchmarks/train_schedule.py once its 8M segment is scored.
cd /Users/misja/PycharmProjects/backgammon-python
until grep -q "^256-256   8M" nets/money/results.txt; do sleep 30; done
pkill -f "nets/money/run.py"; sleep 2; pkill -f "nets/money/train-bin"; sleep 2
echo "--- 256-256: from here on trained by benchmarks/train_schedule.py (lower alpha on plateaus)" >> nets/money/results.txt
exec .venv/bin/python -m benchmarks.train_schedule 256-256 --dir nets/money --objective money --hidden 256,256 \
     --compare "nets/arch/256-256-{m}M-avg.bgnet" >> nets/money/schedule.out 2>&1
