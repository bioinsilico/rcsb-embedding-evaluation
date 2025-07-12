import os

import numpy as np
import pandas as pd

from analysis.analysis_dataset import Depth, is_tp


class EmbeddingDataset:
    def __init__(
            self,
            embedding_path,
            embedding_class_file,
            depth=Depth.scop_family
    ):
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.embedding_path = embedding_path
        self.embedding_class_file = embedding_class_file
        self.depth = depth
        self.load_embedding()
        self.load_class_number()
        super().__init__()

    def load_embedding(self):
        for file in os.listdir(self.embedding_path):
            embedding_id = ".".join(file.split(".")[0:-1])
            v = np.array(list(pd.read_csv(f"{self.embedding_path}/{file}").iloc[:, 0].values))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.embeddings[embedding_id] = v

    def load_class_number(self):
        scop_classes = [(row.strip().split("\t")[0], row.strip().split("\t")[1]) for row in open(self.embedding_class_file)]
        for e_i, d_i in scop_classes:
            self.embeddings_classes[e_i] = d_i
            self.n_classes[e_i] = 0
            for e_j, d_j in scop_classes:
                if e_j == e_i:
                    continue
                if is_tp(self.depth, d_i, d_j):
                    self.n_classes[e_i] += 1

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
