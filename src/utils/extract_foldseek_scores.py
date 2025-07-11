
from operator import itemgetter


def process_score_pairs(
        score_file,
        class_file,
        row_parser,
        depth,
        is_tp,
        fold_fp,

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
        query_scores[e_i].append((e_i, scop_map[e_i], e_j, scop_map[e_j], s))

    print("Calculating sensitivity scores")
    sen_values = []
    for d_i in query_scores:
        if n_classes[scop_map[d_i]] == 0:
            continue
        sort_score = sorted(query_scores[d_i], key=itemgetter(4))
        sort_score.reverse()
        fp = [idx for (idx, (e_i, c_i, e_j, c_j, s)) in enumerate(sort_score) if fold_fp(c_i, c_j)]
        fp_idx = fp[0] if len(fp) > 0 else len(sort_score)
        tp = [(e_i, c_i, e_j, c_j, s) for (e_i, c_i, e_j, c_j, s) in sort_score[0:fp_idx] if is_tp(depth, c_i, c_j)]
        sen = len(tp) / n_classes[scop_map[d_i]]
        sen_values.append(sen)

    sen_values = sorted(sen_values)
    sen_values.reverse()

    return sen_values

