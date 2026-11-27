#!/bin/bash
# Quality check of the faster training (2026-11-27): 4M games at alpha 0.0025 from the 512-256 money
# run's 44M raw net, with offline TD(λ) in f32 (f32) and with f16 move search (f16); compare with the
# old run's 48M segment (online traces), all scored by the same decision benchmark binary.
cd "$(dirname "$0")/../.."
D=nets/speedtest
for v in f32 f16; do
  [ -f $D/b-$v-avg.bgnet ] && continue
  extra=""; [ $v = f16 ] && extra="--search f16"
  /usr/bin/time -p $D/train --variant mixed --threads 12 --games 4000000 --report 1000000 --average-last 500000 \
    --alpha 0.0025 --lambda 0.7 --seed 5252 --bearoff data/bearoff-6x15.bd \
    --resume nets/money/512-256-a0025-48M.bgnet --out $D/b-$v.bgnet $extra > $D/b-$v.log 2>&1
done
for net in nets/money/512-256-a0025-52M-avg.bgnet $D/b-f32-avg.bgnet $D/b-f16-avg.bgnet; do
  echo "== $net"
  $D/decisions --set money --scoring money $net | grep -E "^(attacking|priming|anchoring|racing|all) "
  $D/decisions --set pasko --scoring money $net | grep -E "^all "
done > $D/results-b.txt 2>&1
