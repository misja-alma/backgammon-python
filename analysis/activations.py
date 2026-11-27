"""Which groups of hidden neurons fire together, and for which kinds of positions?

Reads a dump of src/bin/activation_dump.rs (self-play positions with all hidden activations) and
looks at it from two sides:

1. Neuron groups: neurons of a layer clustered by how their activations correlate over all
   positions (hierarchical clustering, distance 1 - r). For each group, the positions where the
   group fires most (top 1% of its mean z-scored activation) are described by backgammon features
   that set them apart from the average position.
2. Position clusters: positions clustered by their layer-2 activation pattern (PCA + k-means),
   without any labels. Each cluster is described by its features, by which neuron groups are most
   active in it, and by a few typical positions.

    .venv-analysis/bin/python -m analysis.activations data/activations/td [--layer 2] [--k 12]

Run it with the analysis environment (.venv-analysis: numpy, scipy, scikit-learn).
"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

RNG = np.random.default_rng(1)
# numpy 2.0 with Apple's Accelerate reports spurious floating-point errors in matmul; the results
# are fine (checked against a float64 SVD).
warnings.filterwarnings("ignore", message=".*encountered in matmul")


# --- loading -----------------------------------------------------------------------------------

def load(prefix):
    info = json.loads(Path(f"{prefix}.json").read_text())
    n, sizes = info["positions"], info["hidden"]
    hidden = np.memmap(f"{prefix}.hidden.f32", dtype="<f4", mode="r", shape=(n, sum(sizes)))
    checkers = np.fromfile(f"{prefix}.positions.u8", dtype=np.uint8).reshape(n, 52).astype(np.int32)
    with open(f"{prefix}.meta.csv") as f:
        rows = list(csv.DictReader(f))
    meta = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    layers, start = [], 0
    for s in sizes:
        layers.append((start, start + s))
        start += s
    return info, hidden, checkers, meta, layers


# --- backgammon features -----------------------------------------------------------------------

def longest_run(made):
    """Longest run of consecutive True along axis 1."""
    best = np.zeros(made.shape[0], dtype=np.int32)
    run = np.zeros(made.shape[0], dtype=np.int32)
    for j in range(made.shape[1]):
        run = np.where(made[:, j], run + 1, 0)
        best = np.maximum(best, run)
    return best


def side_features(c, other):
    """Features of one side; c: (n, 26) counts in its own numbering (0 off, 25 bar)."""
    points = np.arange(26)
    made = c[:, 1:25] >= 2
    f = {
        "pips": (c * points).sum(1),
        "off": c[:, 0],
        "bar": c[:, 25],
        "home made": made[:, :6].sum(1),          # own points 1-6
        "prime": longest_run(made),
        "blots": (c[:, 1:25] == 1).sum(1),
        "back": c[:, 19:26].sum(1),               # checkers in the opponent's home board or on the bar
        "anchor": (c[:, 19:25] >= 2).any(1).astype(int),
        "outfield": c[:, 7:19].sum(1),            # own points 7-18
        "home": c[:, 1:7].sum(1),
        "stacked": np.maximum(c[:, 1:25] - 3, 0).sum(1),   # checkers beyond 3 on a point
    }
    # Blots the opponent can hit directly: opponent checkers behind them within 6 pips (other's
    # point q is this side's 25 - q). A rough proxy for exposure.
    other_mine = np.zeros_like(c[:, 1:25])
    for q in range(1, 26):
        p = 25 - q
        if 1 <= p <= 24:
            other_mine[:, p - 1] += other[:, q]
    exposed = np.zeros(c.shape[0], dtype=np.int32)
    for p in range(1, 25):
        blot = c[:, p] == 1
        attackers = other_mine[:, p:min(p + 6, 24)].sum(1) + (other[:, 25] if p >= 19 else 0)
        exposed += (blot & (attackers > 0)).astype(np.int32)
    f["exposed blots"] = exposed
    return f


def rearmost(c):
    occupied = c[:, 1:26] > 0
    idx = np.where(occupied.any(1), 25 - np.argmax(occupied[:, ::-1], axis=1), 0)
    return idx


def features(checkers, meta):
    me, op = checkers[:, :26], checkers[:, 26:]
    fm, fo = side_features(me, op), side_features(op, me)
    f = {}
    for k in fm:
        f[f"P {k}"] = fm[k]
        f[f"O {k}"] = fo[k]
    f["pip lead"] = fo["pips"] - fm["pips"]   # positive: the player on roll is ahead
    f["contact"] = (rearmost(me) + rearmost(op) > 25).astype(int)
    f["race"] = 1 - f["contact"]
    f["P bearing off"] = (me[:, 7:26].sum(1) == 0).astype(int)
    f["O bearing off"] = (op[:, 7:26].sum(1) == 0).astype(int)
    f["turn"] = meta["turn"]
    f["win prob"] = meta["output"]
    f["nackgammon"] = meta["variant"]
    f["won"] = meta["won"]
    f["gammon end"] = (meta["points"] >= 2).astype(int)
    return {k: np.asarray(v, dtype=np.float64) for k, v in f.items()}


def distinctive(feats, mask, top=8):
    """Features whose mean in `mask` differs most from the overall mean (in overall std units)."""
    out = []
    for k, v in feats.items():
        sd = v.std()
        if sd == 0:
            continue
        out.append(((v[mask].mean() - v.mean()) / sd, k, v[mask].mean(), v.mean()))
    out.sort(key=lambda t: -abs(t[0]))
    return out[:top]


def board(row):
    """Compact text of a position: the player on roll's points (P) and the other's (O), own
    numbering, as point:count; bar and off."""
    def side(c):
        pts = " ".join(f"{p}:{c[p]}" for p in range(24, 0, -1) if c[p])
        extra = (f" bar:{c[25]}" if c[25] else "") + (f" off:{c[0]}" if c[0] else "")
        return pts + extra
    return f"P[{side(row[:26])}]  O[{side(row[26:])}]"


# --- analysis ----------------------------------------------------------------------------------

def neuron_groups(acts, threshold):
    """Clusters of neurons (columns) by activation correlation; dead neurons apart."""
    alive = acts.std(0) > 1e-6
    a = acts[:, alive]
    r = np.corrcoef(a, rowvar=False)
    r = np.nan_to_num(r)
    d = 1 - r
    np.fill_diagonal(d, 0)
    z = linkage(squareform(np.clip(d, 0, 2), checks=False), method="average")
    labels = fcluster(z, t=threshold, criterion="distance")
    ids = np.flatnonzero(alive)
    groups = [ids[labels == g] for g in np.unique(labels)]
    groups.sort(key=len, reverse=True)
    return groups, np.flatnonzero(~alive), r, alive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--layer", type=int, default=2, help="hidden layer to analyze (1 or 2)")
    ap.add_argument("--sample", type=int, default=80000, help="positions used for clustering")
    ap.add_argument("--threshold", type=float, default=0.6, help="neuron grouping distance (1 - r)")
    ap.add_argument("--min-group", type=int, default=4)
    ap.add_argument("--k", type=int, default=12, help="position clusters")
    ap.add_argument("--subset", choices=["all", "contact", "middle"], default="all",
                    help="positions analyzed: all, contact only, or contact with a winning chance of 15-85%%")
    a = ap.parse_args()

    info, hidden, checkers, meta, layers = load(a.prefix)
    lo, hi = layers[a.layer - 1]
    n = hidden.shape[0]
    pool = np.arange(n)
    if a.subset != "all":
        all_feats = features(checkers, meta)
        keep = all_feats["contact"] == 1
        if a.subset == "middle":
            keep &= (all_feats["win prob"] > 0.15) & (all_feats["win prob"] < 0.85)
        pool = np.flatnonzero(keep)
    idx = np.sort(RNG.choice(pool, size=min(a.sample, len(pool)), replace=False))
    acts = np.asarray(hidden[idx, lo:hi], dtype=np.float64)
    feats = features(checkers[idx], {k: v[idx] for k, v in meta.items()})
    chk = checkers[idx]
    print(f"net {info['net']}, layer {a.layer} ({hi - lo} neurons), {len(idx)} of {len(pool)} positions ({a.subset})\n")

    active = (acts > 0).mean(0)
    print(f"neurons active (>0) in: never {np.sum(active == 0)}, <1% {np.sum((active > 0) & (active < .01))}, "
          f"1-10% {np.sum((active >= .01) & (active < .1))}, 10-50% {np.sum((active >= .1) & (active < .5))}, "
          f">=50% {np.sum(active >= .5)}; mean fraction of neurons active per position "
          f"{(acts > 0).mean():.3f}\n")

    # 1. neuron groups
    groups, dead, r, alive = neuron_groups(acts, a.threshold)
    big = [g for g in groups if len(g) >= a.min_group]
    small = sum(len(g) for g in groups if len(g) < a.min_group)
    print(f"=== neuron groups (average linkage, 1 - r < {a.threshold}): {len(big)} groups of >= {a.min_group}, "
          f"{small} neurons in smaller groups, {len(dead)} dead ===\n")
    mu, sd = acts.mean(0), acts.std(0) + 1e-9
    zacts = (acts - mu) / sd
    group_scores = []
    for gi, g in enumerate(big):
        score = zacts[:, g].mean(1)
        group_scores.append(score)
        sub = r[np.ix_(np.searchsorted(np.flatnonzero(alive), g), np.searchsorted(np.flatnonzero(alive), g))]
        within = (sub.sum() - len(g)) / max(1, len(g) * (len(g) - 1))
        top = score >= np.quantile(score, 0.99)
        print(f"--- group {gi}: {len(g)} neurons, mean within-group r {within:.2f}, active in "
              f"{(acts[:, g] > 0).mean():.0%} of positions")
        for z, k, m_in, m_all in distinctive(feats, top):
            print(f"    {k:18s} {m_in:7.2f}  (all {m_all:7.2f}, {z:+.1f} sd)")
        for j in np.argsort(-score)[:3]:
            print(f"    e.g. {board(chk[j])}  win {feats['win prob'][j]:.2f}")
        print()

    # 2. position clusters
    pca = PCA(n_components=min(40, acts.shape[1]), random_state=1).fit(acts)
    x = pca.transform(acts)
    km = KMeans(n_clusters=a.k, n_init=4, random_state=1).fit(x)
    print(f"=== position clusters: k-means (k={a.k}) on {x.shape[1]} principal components "
          f"({pca.explained_variance_ratio_.sum():.0%} of the variance) ===\n")
    gs = np.array(group_scores).T if group_scores else np.zeros((len(idx), 0))
    order = np.argsort(-np.bincount(km.labels_))
    for c in order:
        m = km.labels_ == c
        print(f"--- cluster {c}: {m.mean():.1%} of positions, win prob {feats['win prob'][m].mean():.2f}")
        for z, k, m_in, m_all in distinctive(feats, m):
            print(f"    {k:18s} {m_in:7.2f}  (all {m_all:7.2f}, {z:+.1f} sd)")
        if gs.shape[1]:
            g_mean = gs[m].mean(0)
            best = np.argsort(-g_mean)[:3]
            print("    most active neuron groups: " + ", ".join(f"{b} ({g_mean[b]:+.2f})" for b in best))
        centre = km.cluster_centers_[c]
        near = np.flatnonzero(m)[np.argsort(((x[m] - centre) ** 2).sum(1))[:3]]
        for j in near:
            print(f"    e.g. {board(chk[j])}  win {feats['win prob'][j]:.2f}")
        print()


if __name__ == "__main__":
    main()
