#!/bin/bash
# Scores every finished segment of the f32-576-192 run on the decision benchmark (money and
# Pasko), as soon as its average appears; results in nets/qat/decisions.txt.
cd /Users/misja/PycharmProjects/backgammon-python
OUT=nets/qat/decisions.txt
score() {  # net label
    local m p
    m=$(./target/release/decision_benchmark "$1" | awk '/^all/{print $3, $4}')
    p=$(./target/release/decision_benchmark --set pasko "$1" | awk '/^all/{print $3, $4}')
    g=$(./target/release/decision_benchmark "$1" | awk '/^(attacking|priming|anchoring|racing)/{printf "%s %s  ", substr($1,1,4), $4}')
    printf "%-14s money best %s loss %s | pasko best %s loss %s | %s\n" "$2" ${m% *} ${m#* } ${p% *} ${p#* } "$g" >> $OUT
}
while true; do
    for f in $(ls -tr nets/qat/f32-576-192-*M-avg.bgnet 2>/dev/null); do
        label=$(basename $f -avg.bgnet)
        grep -q "^$label " $OUT 2>/dev/null && continue
        # the average is written when the segment's training ends; wait until it is complete
        sleep 5
        score $f $label
    done
    pgrep -f nets/qat/run.py >/dev/null || break
    sleep 60
done
