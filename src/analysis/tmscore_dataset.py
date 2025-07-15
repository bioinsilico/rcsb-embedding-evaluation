from operator import itemgetter


class TMscoreDataset:

    def __init__(
        self,
        tmscore_file,
        thr,
        alt_scores_file=None,
        row_parser=None,
        reverse=True
    ):

        self.tmscore_file = tmscore_file
        self.tp_thr = thr
        self.reverse = reverse
        self.score_pairs = {}
        self.n_classes = {}
        self.fp_thr = 0.5
        self.n_pos = 0
        self.alt_scores ={}
        if alt_scores_file is not None and row_parser is not None:
            self.parse_alt_scores(alt_scores_file, row_parser)
        self.load_embedding_pairs()

    def parse_alt_scores(self, alt_scores_file, row_parser):
        for r in open(alt_scores_file):
            e_i, e_j, s = row_parser(r.strip().split('\t'))
            if e_i == e_j:
                continue
            if e_i not in self.alt_scores:
                self.alt_scores[e_i] = {}
            self.alt_scores[e_i][e_j] = float(s)

    def parse_score_file(self):
        n_pos = 0
        n_neg = 0
        print(f"Parsing {self.tmscore_file}")
        with open(self.tmscore_file, 'r') as f:
            next(f)
            for r in f:
                e_i, e_j, s, s_p = r.strip().split(',')
                s = float(s)
                s_p = float(s_p)
                if e_i in self.alt_scores and e_j in self.alt_scores[e_i]:
                    s_p = self.alt_scores[e_i][e_j]
                elif len(self.alt_scores) > 0:
                    s_p = 0.0 if self.reverse else 1e6
                if e_i == e_j:
                    continue
                tp = 1 if s >= self.tp_thr else 0
                fp = 1 if s < self.fp_thr else 0
                if e_i not in self.n_classes:
                    self.n_classes[e_i] = 0
                self.n_classes[e_i] += tp
                n_pos += tp
                n_neg += fp
                yield e_i, e_j, s_p, tp, fp
        self.n_pos = n_pos
        print(f"Number of positives: {n_pos}, negatives: {n_neg}")

    def load_embedding_pairs(self):
        for e_i, e_j, s, tp, fp in self.parse_score_file():
            if e_i not in self.score_pairs:
                self.score_pairs[e_i] = []
            self.score_pairs[e_i].append((e_j, s, tp, fp))

    def pairs(self):
        _scores = self.__get_score_pairs()
        _scores = sorted(_scores, key=itemgetter(0))
        if self.reverse:
            _scores.reverse()
        return [(tp, fp) for s, tp, fp in _scores]

    def pairs_len(self):
        return len(self.__get_score_pairs())

    def __get_score_pairs(self):
        return [(s, tp, fp) for e_i ,_values in self.score_pairs.items() if self.n_classes[e_i] > 0 for (e_j, s, tp, fp) in _values if self.n_classes[e_j] > 0]

    def get_n_classes(self, name):
        return self.n_classes[name]

    def get_n_pos(self):
        return self.n_pos

    def domains(self):
        return [e_i for e_i in self.score_pairs.keys() if self.n_classes[e_i] > 0]

    def get_domain_scores(self, e_i):
        _scores = sorted([(e_j, s, tp, fp) for (e_j, s, tp, fp) in self.score_pairs[e_i] if self.n_classes[e_j] > 0], key=itemgetter(1))
        if self.reverse:
            _scores.reverse()
        return _scores