#!/bin/bash
# 40k money games per variant against bgsage for the f16 copy of the speed check's by-product net.
cd "$(dirname "$0")/../.."
for v in backgammon nackgammon; do
  .venv/bin/python -m benchmarks.strength --net nets/speedtest/b-f32-avg-f16.bgnet --variant $v --games 40000 \
    --bearoff data/bearoff-6x15.bd --note "speedtest b-f32-avg f16" --money 2>&1 | grep -E "Points per game"
done
