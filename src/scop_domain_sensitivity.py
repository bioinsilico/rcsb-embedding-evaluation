"""
Sensitivity-to-first-false-positive vs fraction-of-queries for a SCOP
single-domain benchmark.

All domains belong to a single set: each domain is compared against every
other domain in the set.

Input files:
  - Domain classes TSV (no header): domain_id <tab> scop_family (e.g. a.4.5.3)
  - Score file (no header): domain_id_1 <tab> domain_id_2 <tab> score

Per-hit classification (comparing query family to subject family):
  - same family (all 4 levels match)       -> TP
  - same fold but different family          -> IGNORE (skipped)
  - different fold                          -> FP

Self-matches (same domain_id) are removed before scoring.

Per-query metric:
  walk the ranked list (descending score), count TPs until the first FP;
  sensitivity = TPs_before_first_FP / (total_domains_sharing_family - 1)

Queries with zero possible TPs are dropped.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def fold_of(fam):
    """'a.4.5.5' -> 'a.4'."""
    a, b, _, _ = fam.split(".")
    return f"{a}.{b}"


def classify(query_fam, subject_fam):
    """Return 'TP', 'FP', or 'IGNORE'."""
    if query_fam == subject_fam:
        return "TP"
    if fold_of(query_fam) == fold_of(subject_fam):
        return "IGNORE"
    return "FP"


def load_domain_classes(path):
    """domain_id -> SCOP family string 'a.b.c.d'."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            out[parts[0]] = parts[1]
    return out


def load_scores(path):
    """Yield (domain_id_1, domain_id_2, score) from a 3-column whitespace file."""
    rows_by_query = defaultdict(list)
    with open(path) as fh:
        for order, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            query, target = parts[0], parts[1]
            try:
                score = float(parts[2])
            except ValueError:
                continue
            rows_by_query[query].append((-score, order, target))
    for q, rows in rows_by_query.items():
        rows.sort()
        yield q, [target for *_, target in rows]


def compute_sensitivities(scores_path, domain_classes):
    # Count domains per family for denominators.
    fam_counts = defaultdict(int)
    for fam in domain_classes.values():
        fam_counts[fam] += 1

    sensitivities = {}
    for query, matches in load_scores(scores_path):
        if query not in domain_classes:
            continue
        q_fam = domain_classes[query]

        # Denominator: domains sharing the family minus the query itself.
        possible = fam_counts[q_fam] - 1
        if possible <= 0:
            continue

        tp_before_fp = 0
        for target in matches:
            if target == query:
                continue  # skip self-match
            if target not in domain_classes:
                continue
            label = classify(q_fam, domain_classes[target])
            if label == "TP":
                tp_before_fp += 1
            elif label == "IGNORE":
                continue
            else:  # FP
                break

        sensitivities[query] = tp_before_fp / possible

    return sensitivities


def plot(series, out_path):
    """series: list of (label, {query: sensitivity})."""
    series = [(lbl, s) for lbl, s in series if s]
    if not series:
        raise SystemExit("No queries with non-zero possible positives; nothing to plot.")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sensitivities in series:
        values = sorted(sensitivities.values(), reverse=True)
        n = len(values)
        x = [(i + 1) / n for i in range(n)]
        ax.plot(x, values, linewidth=1.5, label=f"{label} (n={n})")

    ax.set_xlabel("Fraction of queries")
    ax.set_ylabel("Sensitivity to first false positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--classes",
        required=True,
        type=Path,
        help="TSV with two columns: domain_id and SCOP family (a.b.c.d).",
    )
    ap.add_argument(
        "--scores",
        required=True,
        action="append",
        type=lambda s: tuple(s.split("=", 1)) if "=" in s else (None, s),
        help=(
            "Score file (3-col: query, target, score; higher is better). "
            "Repeatable. Use 'label=path' to set a legend label."
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("../img/scop-domain-bench.png"))
    ap.add_argument(
        "--dump-csv",
        type=Path,
        help="Optional: write per-query sensitivities to CSV.",
    )
    args = ap.parse_args()

    domain_classes = load_domain_classes(args.classes)

    series = []
    for label, path in args.scores:
        sensitivities = compute_sensitivities(Path(path), domain_classes)
        series.append((label or Path(path).stem, sensitivities))

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Method", "Query", "Sensitivity"])
            for label, sens in series:
                for q, s in sorted(sens.items(), key=lambda kv: -kv[1]):
                    w.writerow([label, q, f"{s:.6f}"])

    plot(series, args.out)


if __name__ == "__main__":
    main()
