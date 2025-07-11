
import pandas as pd


def get_tp(depth):
    def __get_class(class_id):
        return ".".join(class_id.split(".")[0:depth]), class_id.split(".")[depth]
    return __get_class


def is_tp(depth, d_i, d_j):
    if depth == Depth.scop_family:
        return d_i == d_j
    else:
        c_i, x_i = get_tp(depth)(d_i)
        c_j, x_j = get_tp(depth)(d_j)
        return c_i == c_j and x_i != x_j

def is_fp(d_i, d_j):
    c_tree = d_i.split(".")
    c_anchor_tree = d_j.split(".")
    if c_tree[0] == c_anchor_tree[0] and c_tree[1] == c_anchor_tree[1]:
        return False
    return True

class Depth:
    scop_class = 1
    scop_fold = 2
    scop_super_family = 3
    scop_family = 4


def depth_name(depth):
    if depth == Depth.scop_family:
        return "SCOPe Family"
    if depth == Depth.scop_super_family:
        return "SCOPe Super Family"
    if depth == Depth.scop_fold:
        return "SCOPe Fold"
    if depth == Depth.scop_class:
        return "SCOPe Class"
    raise ValueError("Invalid SCOPe depth")


class AnalysisDataset:
    def __init__(
            self,
            score_file,
            score_row_parser,
            dom_class_file,
            depth=Depth.scop_family,
            score_reverse=False
    ):
        self.score_pairs = pd.DataFrame()
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.n_fam = {}
        self.n_sfam = {}
        self.n_fold = {}
        self.score_file = score_file
        self.score_row_parser = score_row_parser
        self.dom_class_file = dom_class_file
        self.n_pos = 0
        self.w_pos = 0
        self.depth = depth
        self.load_class_number()
        self.load_embedding_pairs(score_reverse=score_reverse)

    def load_class_number(self):
        scop_classes = [(row.strip().split("\t")[0], row.strip().split("\t")[1]) for row in open(self.dom_class_file)]
        for e_i, d_i in scop_classes:
            self.embeddings_classes[e_i] = d_i
            self.n_classes[e_i] = 0
            self.n_fam[e_i] = 0
            self.n_sfam[e_i] = 0
            self.n_fold[e_i] = 0
            for e_j, d_j in scop_classes:
                if e_j == e_i:
                    continue
                if is_tp(self.depth, d_i, d_j):
                    self.n_classes[e_i] += 1
                if is_tp(Depth.scop_family, d_i, d_j):
                    self.n_fam[e_i] += 1
                if is_tp(Depth.scop_super_family, d_i, d_j):
                    self.n_sfam[e_i] += 1
                if is_tp(Depth.scop_fold, d_i, d_j):
                    self.n_fold[e_i] += 1

    def parse_score_file(self):
        n_pos = 0
        n_neg = 0
        print(f"Parsing {self.score_file}")
        for r in open(self.score_file, 'r'):
            e_i, e_j, s = self.score_row_parser(r.strip().split('\t'))
            if e_i == e_j:
                continue
            if self.n_classes[e_i] == 0 or self.n_classes[e_j] == 0:
                continue
            d_i = self.embeddings_classes[e_i]
            d_j = self.embeddings_classes[e_j]
            tp = 0
            if is_tp(self.depth, d_i, d_j):
                tp = self.tp_weight(e_i)
            fp = 0
            if is_fp(d_i, d_j):
                fp = self.fp_weight(e_i)
            n_pos += tp
            n_neg += fp
            yield s, tp, fp
        self.n_pos = n_pos
        print(f"Number of positives: {n_pos}, negatives: {n_neg}")

    def load_embedding_pairs(self, score_reverse=False):
        self.score_pairs = pd.DataFrame(self.parse_score_file(), columns=['score', 'tp', 'fp'])
        self.score_pairs = self.score_pairs.sort_values(by='score', ascending=score_reverse)

    def reload_embedding_pairs(self, score_file, score_row_parser, score_reverse=False):
        self.score_file = score_file
        self.score_row_parser = score_row_parser
        self.load_embedding_pairs(score_reverse=score_reverse)

    def pairs(self):
        return self.score_pairs[['tp', 'fp']].itertuples(index=False, name=None)

    def pairs_len(self):
        return len(self.score_pairs)

    def domains(self, n=1):
        for embedding_id in [
            e for e in self.embeddings
            if e in self.embeddings_classes and self.n_classes[e] >= n
        ]:
            yield embedding_id, self.embeddings[embedding_id]

    def get_class(self, dom):
        if dom not in self.embeddings_classes:
            raise Exception(f"Undefined class for domain: {dom}")
        return self.embeddings_classes[dom]

    def get_n_classes(self, name):
        return self.n_classes[name]

    def get_n_pos(self):
        return self.n_pos

    def tp_weight(self, e_i):
        if self.depth == Depth.scop_family:
            return 1 / self.n_fam[e_i]
        if self.depth == Depth.scop_super_family:
            return 1 / self.n_sfam[e_i]
        if self.depth == Depth.scop_fold:
            return 1 / self.n_fold[e_i]
        raise Exception(f"Unknown depth {self.depth}")

    def fp_weight(self, e_i):
        norm = self.n_fold[e_i] + self.n_sfam[e_i] + self.n_fam[e_i]
        norm = 1 if norm == 0 else norm
        return 1 / norm
