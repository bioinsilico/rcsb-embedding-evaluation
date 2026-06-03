"""
Sensitivity-to-first-false-positive vs fraction-of-queries for a SCOP-annotated
sequence-search benchmark.

Per-hit classification (subject vs query SCOP family a.b.c.d):
  - unannotated subject                                   -> FP
  - any of the subject's SCOPs matches at all 4 levels    -> TP
  - any of the subject's SCOPs matches at 2 levels only   -> IGNORE (skipped)
  - otherwise                                             -> FP

Self-matches (same UniProt accession on both sides) are removed before scoring.

Per-query metric:
  walk the ranked list, count TPs until (but not including) the first FP;
  sensitivity = TPs_before_first_FP / total_annotated_subjects_sharing_family
  (denominator excludes the query's own UniProt).

Queries with zero possible TPs in the database are dropped from the plot.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_fasta_headers(path):
    """Yield (id_token, rest_of_header) for each '>' line."""
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].rstrip("\n")
            parts = header.split(None, 1)
            yield parts[0], (parts[1] if len(parts) > 1 else "")


def scop_family(token):
    """First whitespace-separated token of the header rest, expected like 'a.4.5.5'."""
    token = token.strip()
    if not token:
        return None
    first = token.split()[0]
    if first.count(".") != 3:
        return None
    return first


def load_query_scop(query_fasta):
    """query_id -> SCOP family string 'a.b.c.d'."""
    out = {}
    for qid, rest in parse_fasta_headers(query_fasta):
        fam = scop_family(rest)
        if fam is not None:
            out[qid] = fam
    return out


def load_subject_scop(subject_fasta):
    """
    subject_id -> set of SCOP families (only populated for annotated_* entries).
    Unannotated entries are recorded with an empty set so we can distinguish
    'known but unannotated' from 'unknown id'.
    """
    out = defaultdict(set)
    for sid, rest in parse_fasta_headers(subject_fasta):
        if sid.startswith("annotated_"):
            fam = scop_family(rest)
            if fam is not None:
                out[sid].add(fam)
            else:
                out[sid]  # ensure key exists
        else:
            out[sid]  # known but unannotated
    return dict(out)


def uniprot_of_query(qid):
    """'P30340_1' -> 'P30340'. Strips a trailing '_<digits>' domain suffix if present."""
    if "_" in qid:
        head, tail = qid.rsplit("_", 1)
        if tail.isdigit():
            return head
    return qid


def uniprot_of_subject(sid):
    """'annotated_P30340' -> 'P30340'; otherwise return sid unchanged."""
    if sid.startswith("annotated_"):
        return sid[len("annotated_"):]
    return sid


def fold_of(fam):
    """'a.4.5.5' -> 'a.4'."""
    a, b, _, _ = fam.split(".")
    return f"{a}.{b}"


def classify(query_fam, subject_fams):
    """Return 'TP', 'FP', or 'IGNORE'."""
    if not subject_fams:
        return "FP"
    if query_fam in subject_fams:
        return "TP"
    q_fold = fold_of(query_fam)
    if any(fold_of(f) == q_fold for f in subject_fams):
        return "IGNORE"
    return "FP"


def total_positives_per_family(subject_scop):
    """SCOP family -> set of subject UniProt accessions annotated with it."""
    fam_to_uniprots = defaultdict(set)
    for sid, fams in subject_scop.items():
        if not fams:
            continue
        up = uniprot_of_subject(sid)
        for fam in fams:
            fam_to_uniprots[fam].add(up)
    return fam_to_uniprots


def read_results_csv(path):
    """CSV with header 'Query,Rank,Match,Score'. Best hit = lowest Rank."""
    rows_by_query = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_by_query[row["Query"]].append(
                (int(row["Rank"]), row["Match"])
            )
    for q, rows in rows_by_query.items():
        rows.sort(key=lambda r: r[0])
        yield q, [match for _, match in rows]


def read_results_mmseqs(path):
    """
    mmseqs2 / BLAST tabular output (-outfmt 6), 12 cols, no header:
      query, subject, pident/fident, length/alnlen, mismatch, gapopen,
      qstart, qend, sstart/tstart, send/tend, evalue, bitscore/bits
    Rank within a query = ascending evalue (then descending bitscore, then
    file order) — works for both BLAST and mmseqs since the column layout
    is identical.
    """
    rows_by_query = defaultdict(list)
    with open(path) as fh:
        for order, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            query, target = parts[0], parts[1]
            evalue = float(parts[10])
            bits = float(parts[11])
            rows_by_query[query].append((evalue, -bits, order, target))
    for q, rows in rows_by_query.items():
        rows.sort()
        yield q, [target for *_, target in rows]


def read_results_tsv(path):
    """
    4-column TSV (no header): query, subject, score1, score2.
    Higher score1 is better; score2 used as tiebreaker (also higher = better);
    then file order. Whitespace separator (handles aligned columns).
    """
    rows_by_query = defaultdict(list)
    with open(path) as fh:
        for order, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            query, target = parts[0], parts[1]
            try:
                s1 = float(parts[2])
                s2 = float(parts[3])
            except ValueError:
                continue
            rows_by_query[query].append((-s1, -s2, order, target))
    for q, rows in rows_by_query.items():
        rows.sort()
        yield q, [target for *_, target in rows]


def read_results(path, fmt):
    if fmt == "csv":
        return read_results_csv(path)
    if fmt in ("mmseqs", "blast"):
        # BLAST -outfmt 6 and mmseqs default tabular share the same 12 columns.
        return read_results_mmseqs(path)
    if fmt == "tsv":
        return read_results_tsv(path)
    raise ValueError(f"Unknown format: {fmt}")


def compute_sensitivities(results_path, fmt, query_scop, subject_scop):
    fam_to_uniprots = total_positives_per_family(subject_scop)
    sensitivities = {}

    for query, matches in read_results(results_path, fmt):
        if query not in query_scop:
            continue
        q_fam = query_scop[query]
        q_up = uniprot_of_query(query)

        # Denominator: annotated subjects sharing the family, minus the query itself.
        possible = fam_to_uniprots.get(q_fam, set()) - {q_up}
        if not possible:
            continue

        tp_before_fp = 0
        for match in matches:
            if uniprot_of_subject(match) == q_up:
                continue  # exclude self-match
            subject_fams = subject_scop.get(match, set())
            label = classify(q_fam, subject_fams)
            if label == "TP":
                tp_before_fp += 1
            elif label == "IGNORE":
                continue
            else:  # FP
                break

        sensitivities[query] = tp_before_fp / len(possible)

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--results",
        required=True,
        action="append",
        type=lambda s: tuple(s.split("=", 1)) if "=" in s else (None, s),
        help=(
            "Result file. Repeatable to overlay multiple methods. "
            "Use 'label=path' to set a legend label (e.g. --results mmseqs=q.tsv --results emb=q.csv). "
            "Pair with --format positionally in the same order."
        ),
    )
    ap.add_argument(
        "--format",
        required=True,
        action="append",
        choices=["csv", "mmseqs", "blast", "tsv"],
        help=(
            "Format for each --results in order. "
            "csv = Query,Rank,Match,Score header; "
            "mmseqs/blast = 12-col TSV (-outfmt 6 style, evalue ascending); "
            "tsv = 4-col TSV (query, subject, score1, score2; score1 descending)."
        ),
    )
    ap.add_argument("--query-fasta", required=True, type=Path)
    ap.add_argument("--subject-fasta", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("../img/qs-sequence-bench.png"))
    ap.add_argument("--dump-csv", type=Path, help="Optional: write per-query sensitivities (last input only) to CSV")
    args = ap.parse_args()

    if len(args.results) != len(args.format):
        ap.error("--results and --format must be provided the same number of times, in matching order")

    query_scop = load_query_scop(args.query_fasta)
    subject_scop = load_subject_scop(args.subject_fasta)

    series = []
    sensitivities = None
    for (label, path), fmt in zip(args.results, args.format):
        sensitivities = compute_sensitivities(Path(path), fmt, query_scop, subject_scop)
        series.append((label or f"{fmt}:{Path(path).name}", sensitivities))

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
