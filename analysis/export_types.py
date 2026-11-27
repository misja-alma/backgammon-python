"""Export the game types of the mixture of experts (src/moe.rs) from an activation dump.

A game type is a set of layer-2 neuron groups of the reference net (found by
analysis/activations.py on contact positions with a winning chance of 15-85%: --subset middle,
seed 1, 80000 positions, grouping distance 0.6). A position's score for a type is the mean
z-scored activation of the type's neurons; it belongs to the type when it's a contact position
and the score reaches the type's threshold, the --quantile of the scores of all contact positions
in the dump (so each type takes the top 1 - quantile of them; a position can have several types).
Positions with no type belong to the generic expert.

    .venv-analysis/bin/python -m analysis.export_types data/activations/td nets/moe/types.txt

Output (text, read by src/moe.rs):
    reference <net path>
    layer 2
    type <name> <threshold> <n>
    neurons <n indices>
    mu <n floats>
    sd <n floats>
"""
import argparse
from pathlib import Path

import numpy as np

from analysis.activations import RNG, features, load, neuron_groups

# Game type -> neuron groups of the middle-game run (see neural_net_evaluator.md, 2026-11-26).
TYPES = [
    ("holding-ahead", [0, 9]),          # P primes, O holds an anchor
    ("holding-behind", [5, 14]),        # P holds an anchor against O's prime
    ("blitz-attacker", [1, 11]),        # O on the bar against P's strong board
    ("blitz-defender", [12]),           # P on the bar against O's strong board
    ("prime-trapper", [2]),             # P has a 5-6 point prime, O stuck behind it
    ("prime-trapped", [4]),             # P stuck behind O's prime
    ("contact-bearoff-ahead", [3, 10]),  # P bears off, O still has back checkers
    ("contact-bearoff-behind", [7]),    # O bears off, P still has back checkers
    ("back-game", [6]),                 # P deep in a back game
    ("escaped-opponent", [13, 15]),     # O has escaped all back checkers
    ("late-hit", [8]),                  # P nearly borne off, hit and on the bar
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--quantile", type=float, default=0.92)
    a = ap.parse_args()

    info, hidden, checkers, meta, layers = load(a.prefix)
    lo, hi = layers[1]
    feats = features(checkers, meta)
    contact = feats["contact"] == 1
    middle = np.flatnonzero(contact & (feats["win prob"] > 0.15) & (feats["win prob"] < 0.85))
    idx = np.sort(RNG.choice(middle, size=min(80000, len(middle)), replace=False))   # as activations.py
    acts = np.asarray(hidden[idx, lo:hi], dtype=np.float64)
    groups, _, _, _ = neuron_groups(acts, 0.6)
    groups = [g for g in groups if len(g) >= 4]
    mu, sd = acts.mean(0), acts.std(0) + 1e-9

    all_contact = np.flatnonzero(contact)
    sample = np.sort(RNG.choice(all_contact, size=min(150000, len(all_contact)), replace=False))
    contact_acts = np.asarray(hidden[sample, lo:hi], dtype=np.float64)
    lines = [f"reference {info['net']}", "layer 2"]
    member = np.zeros((len(sample), len(TYPES)), dtype=bool)
    for t, (name, gids) in enumerate(TYPES):
        neurons = np.sort(np.concatenate([groups[g] for g in gids]))
        score = ((contact_acts[:, neurons] - mu[neurons]) / sd[neurons]).mean(1)
        threshold = float(np.quantile(score, a.quantile))
        member[:, t] = score >= threshold
        lines += [f"type {name} {threshold:.6f} {len(neurons)}",
                  "neurons " + " ".join(str(int(n)) for n in neurons),
                  "mu " + " ".join(f"{mu[n]:.6f}" for n in neurons),
                  "sd " + " ".join(f"{sd[n]:.6f}" for n in neurons)]
        print(f"{name:24s} {len(neurons):3d} neurons, threshold {threshold:+.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    k = member.sum(1)
    race = 1 - contact.mean()
    print(f"\ncontact positions ({contact.mean():.0%} of all): no type {np.mean(k == 0):.0%}, one {np.mean(k == 1):.0%}, "
          f"two {np.mean(k == 2):.0%}, more {np.mean(k > 2):.0%}; races ({race:.0%}) are generic")
    overlap = (member[:, :, None] & member[:, None, :]).sum(0) / member.sum(0)[:, None]
    print("largest overlaps (share of the row type's positions also in the column type):")
    pairs = [(overlap[i, j], TYPES[i][0], TYPES[j][0]) for i in range(len(TYPES)) for j in range(len(TYPES)) if i != j]
    for v, x, y in sorted(pairs, reverse=True)[:8]:
        print(f"    {x:24s} -> {y:24s} {v:.0%}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
