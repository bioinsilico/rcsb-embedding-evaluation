import os
import pandas as pd
import numpy as np


class CorrelationDataset:
    def __init__(
            self,
            embedding_path,
            tm_score_file
    ):
        self.embedding_pairs = {}
        self.embeddings = {}
        self.embeddings_idx = {}
        self.load_embedding(embedding_path)
        self.load_tm_score(tm_score_file)
        super().__init__()

    def load_embedding(self, embedding_path):
        n = 0
        for file in os.listdir(embedding_path):
            embedding_id = ".".join(file.split(".")[0:-1])
            v = np.array(list(pd.read_csv(f"{embedding_path}/{file}", header=None).iloc[:, 0].values))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.embeddings_idx[embedding_id] = n
            self.embeddings[n] = v
            n += 1
        print(f"Number of embeddings: {len(self.embeddings_idx)}")

    def load_tm_score(self, tm_score_file):
        for row in open(tm_score_file):
            [d_i, d_j, s] = row.strip().split(",")
            if d_i not in self.embeddings_idx or d_j not in self.embeddings_idx:
                continue
            d_i = self.embeddings_idx[d_i]
            d_j = self.embeddings_idx[d_j]
            if d_i not in self.embedding_pairs:
                self.embedding_pairs[d_i] = {}
            if d_j not in self.embedding_pairs:
                self.embedding_pairs[d_j] = {}
            self.embedding_pairs[d_i][d_j] = float(s)
            self.embedding_pairs[d_j][d_i] = float(s)
        print(f"Number of domains available for tm-score: {len(self.embedding_pairs)}")

    def domains(self):
        for embedding_id in self.embeddings:
            yield embedding_id, self.embeddings[embedding_id]

    def get_domain_scores(self, d_i):
        if d_i not in self.embedding_pairs:
            return []
        for d_j in self.embedding_pairs[d_i]:
            if d_j in self.embeddings:
                yield d_j, self.embeddings[d_j], self.embedding_pairs[d_i][d_j]
