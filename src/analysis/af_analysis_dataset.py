

from analysis.tmscore_dataset import TMscoreDataset



def get_tp(depth):
    def __get_class(class_id):
        return ".".join(class_id.split(".")[0:depth])
    return __get_class

def is_tp(depth, d_i, d_j):
    _is_tp = get_tp(depth)
    m_d_i = [_is_tp(_d_i) for _d_i in d_i.split(",")]
    m_d_j = [_is_tp(_d_j) for _d_j in d_j.split(",")]
    if ",".join(m_d_i) == ",".join(m_d_j):
            return True
    return False

def is_fp(d_i, d_j):
    m_d_i = [".".join(_d_i.split(".")[0:2]) for _d_i in d_i.split(",")]
    m_d_j = [".".join(_d_j.split(".")[0:2]) for _d_j in d_j.split(",")]
    for _d_i in m_d_i:
        if _d_i in m_d_j:
            return False
    return True

class Depth:
    cath_class = 1
    cath_archi = 2
    cath_topol = 3

class AfCathAnalysisDataset(TMscoreDataset):
    def __init__(
            self,
            score_file,
            score_row_parser,
            dom_class_file,
            depth=Depth.cath_topol,
            score_reverse=True
    ):
        self.score_pairs = {}
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.score_file = score_file
        self.score_row_parser = score_row_parser
        self.dom_class_file = dom_class_file
        self.n_pos = 0
        self.depth = depth
        self.reverse = score_reverse
        self.load_class_number()
        self.load_embedding_pairs()

    def load_class_number(self):
        print(f"Parsing {self.dom_class_file}")
        cath_classes = [(row.strip().split("\t")[0], row.strip().split("\t")[1]) for row in open(self.dom_class_file)]
        n_pos = 0
        for e_i, d_i in cath_classes:
            self.embeddings_classes[e_i] = d_i
            self.n_classes[e_i] = 0
            for e_j, d_j in cath_classes:
                if e_j == e_i:
                    continue
                if is_tp(self.depth, d_i, d_j):
                    self.n_classes[e_i] += 1
                    n_pos += 1
        print(f"Number of positives pairs: {n_pos} number of domains: {len(self.n_classes)}")

    def parse_score_file(self):
        n_pos = 0
        n_neg = 0
        print(f"Parsing {self.score_file}")
        for r in open(self.score_file, 'r'):
            e_i, e_j, s = self.score_row_parser(r.strip().split('\t'))
            s = float(s)
            if e_i == e_j:
                continue
            if e_i not in self.n_classes or e_j not in self.n_classes:
                continue
            if self.n_classes[e_i] == 0 or self.n_classes[e_j] == 0:
                continue
            d_i = self.embeddings_classes[e_i]
            d_j = self.embeddings_classes[e_j]
            tp = 1 if is_tp(self.depth, d_i, d_j) else 0
            fp = 1 if is_fp(d_i, d_j) else 0
            #if fp ==1 and s > 0.8:
            #    print(e_i, e_j)
            n_pos += tp
            n_neg += fp
            yield e_i, e_j, s, tp, fp
        self.n_pos = n_pos
        print(f"Number of positives: {n_pos}, negatives: {n_neg}")
