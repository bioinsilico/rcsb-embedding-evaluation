from operator import itemgetter


class LDDTDataset:

    def __init__(self, lddt_file, score_file, row_parser, thr, reverse=True):

        self.lddt_file = lddt_file
        self.lddt_pairs = {}
        self.score_pairs = {}
        self.score_file = score_file
        self.row_parser = row_parser
        self.tp_thr = thr
        self.reverse = reverse
        self.fp_thr = 0.2
        self.n_pos = 0
        self.n_classes = {}

        self.load_lddt_file()
        self.load_embedding_pairs()

    def load_lddt_file(self):
        print(f"Parsing {self.lddt_file}")
        with open(self.lddt_file, 'r') as f:
            for r in f:
                e_i, e_j, s = r.strip().replace("-", ".").split('\t')
                s = float(s)
                self.lddt_pairs[(e_i, e_j)] = s
                self.lddt_pairs[(e_j, e_i)] = s

    def parse_score_file(self):
        n_pos = 0
        n_neg = 0
        print(f"Parsing {self.score_file}")
        with open(self.score_file, 'r') as f:
            for r in f:
                e_i, e_j, s = self.row_parser(r.strip().split('\t'))
                if (e_i, e_j) not in self.lddt_pairs:
                    continue
                l_s = self.lddt_pairs[(e_i, e_j)]
                tp = 1 if l_s >= self.tp_thr else 0
                fp = 1 if l_s < self.fp_thr else 0
                if e_i not in self.n_classes:
                    self.n_classes[e_i] = 0
                self.n_classes[e_i] += tp
                n_pos += tp
                n_neg += fp
                yield e_i, e_j, s, tp, fp

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