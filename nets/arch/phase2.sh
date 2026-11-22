#!/bin/bash
# After the old sweep has been stopped (756 20M done): benchmark the plateaued nets again with the
# bearoff database (40k games per variant, seed 1), then continue the sweep (512-256) with the
# database in training and benchmarks.
cd /Users/misja/PycharmProjects/backgammon-python
export PYGAME_HIDE_SUPPORT_PROMPT=1
until grep -q "^756 .*20M" nets/arch/results.txt && ! pgrep -f "nets/arch/run.py" >/dev/null; do sleep 30; done
sleep 10
for spec in "nets/flags/mixed-30M-avg.bgnet:128-128 C 30M (baseline)" \
            "nets/arch/256-128-24M-avg.bgnet:256-128 24M" \
            "nets/arch/512-128-24M-avg.bgnet:512-128 24M" \
            "nets/arch/256-256-256-36M-avg.bgnet:256-256-256 36M" \
            "nets/arch/256-256-44M-avg.bgnet:256-256 44M (default)"; do
  net=${spec%%:*}; label=${spec#*:}
  bg=$(.venv/bin/python -m benchmarks.strength --net $net --variant backgammon --games 40000 --bearoff data/bearoff-6x15.bd \
       --note "rebench with bearoff db: $label" 2>&1 | grep -o "Win rate [0-9.]*" | grep -o "[0-9.]*$")
  nack=$(.venv/bin/python -m benchmarks.strength --net $net --variant nackgammon --games 40000 --bearoff data/bearoff-6x15.bd \
       --note "rebench with bearoff db: $label" 2>&1 | grep -o "Win rate [0-9.]*" | grep -o "[0-9.]*$")
  mean=$(python3 -c "print(f'{($bg + $nack) / 2:.2f}')")
  echo "with bearoff db: $label: bg $bg  nack $nack  mean $mean" >> nets/arch/results.txt
done
.venv/bin/python nets/arch/run.py >> nets/arch/run.out 2>&1
