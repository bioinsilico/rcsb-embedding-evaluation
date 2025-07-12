
from operator import itemgetter

from tqdm import tqdm

from analysis.analysis_dataset import is_fp, is_tp


def qs_scores(
        score_file,
        class_file,
        row_parser,
        depth,
        reverse=True
):
    scop_map = {}
    n_classes = {}

    print(f"Loading SCOPe map {class_file}")
    for row in open(class_file):
        r = row.strip().split("\t")
        scop_map[r[0]] = r[1]

    print(f"Calculating number of class positives")
    dom_list = scop_map.keys()
    for c_i, d_i in set([(scop_map[d_i], d_i) for d_i in dom_list]):
        if c_i in n_classes:
            continue
        n_classes[c_i] = 0
        for c_j, d_j in [(scop_map[d_j], d_j) for d_j in dom_list]:
            if d_i == d_j:
                continue
            if is_tp(depth, c_i, c_j):
                n_classes[c_i] += 1

    print(f"Loading results {score_file}")
    query_scores = {}
    for row in open(score_file):
        e_i, e_j, s = row_parser(row.strip().split("\t"))
        if n_classes[scop_map[e_i]] == 0:
            continue
        if e_i not in query_scores:
            query_scores[e_i] = []
        if e_j == e_i:
            continue
        d_i = scop_map[e_i]
        d_j = scop_map[e_j]
        tp = is_tp(depth, d_i, d_j)
        fp = is_fp(d_i, d_j)
        query_scores[e_i].append((s, tp, fp))

    print("Calculating sensitivity scores")
    sen_values = []
    with tqdm(total=len(query_scores), desc="Domain Pairs", unit="pair") as pbar:
        for e_i in query_scores:
            if n_classes[scop_map[e_i]] == 0:
                continue
            sort_score = sorted(query_scores[e_i], key=itemgetter(0))
            if reverse:
                sort_score.reverse()
            fp = [idx for (idx, (s, tp, fp)) in enumerate(sort_score) if fp]
            fp_idx = fp[0] if len(fp) > 0 else len(sort_score)
            tp = [s for (s, tp, fp) in sort_score[0:fp_idx] if tp]
            sen = len(tp) / n_classes[scop_map[e_i]]
            sen_values.append(sen)
            pbar.update(1)

    sen_values = sorted(sen_values)
    sen_values.reverse()

    return sen_values

