import os
import pandas as pd
import numpy as np


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
    if depth == Depth.scop_family:
        return "SCOPe Class"
    raise ValueError("Invalid SCOPe depth")


class AnalysisDataset:
    def __init__(
            self,
            embedding_path,
            embedding_class_file,
            depth=Depth.scop_family
    ):
        self.embedding_pairs = []
        self.embeddings = {}
        self.embeddings_classes = {}
        self.n_classes = {}
        self.embedding_path = embedding_path
        self.embedding_classe_file = embedding_class_file
        self.depth = depth
        self.get_tp = get_tp(self.depth)
        self.load_class_number()
        self.load_embedding()
        self.load_embedding_pairs()
        super().__init__()

    def load_embedding_int8(self):
        min_val = 1000
        max_val = 0
        for file in os.listdir(self.embedding_path):
            embedding_id = ".".join(file.split(".")[0:-1])
            v = np.array(list(pd.read_csv(f"{self.embedding_path}/{file}", header=None).iloc[:, 0].values))
            if v.min() < min_val:
                min_val = v.min()
            if v.max() > max_val:
                max_val = v.max()
            self.embeddings[embedding_id] = v
        for embedding_id in self.embeddings:
            v = self.embeddings[embedding_id]
            v = np.round((v - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.embeddings[embedding_id] = v

    def load_embedding(self):
        for embedding_id in self.embeddings_classes.keys():
            v = np.array(list(pd.read_csv(f"{self.embedding_path}/{embedding_id}.csv").iloc[:, 0].values))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.embeddings[embedding_id] = v

    def load_class_number(self):
        scop_classes = [(row.strip().split("\t")[0], row.strip().split("\t")[1]) for row in open(self.embedding_classe_file)]
        for e_i, d_i in scop_classes:
            self.embeddings_classes[e_i] = d_i
            self.n_classes[e_i] = 0
            for e_j, d_j in scop_classes:
                if e_j == e_i:
                    continue
                if is_tp(self.depth, d_i, d_j):
                    self.n_classes[e_i] += 1

    def load_embedding_pairs(self):
        ids = list(self.embeddings.keys())
        n_pos = 0
        n_neg = 0
        while len(ids) > 0:
            embedding_i = ids.pop()
            for embedding_j in ids:
                pred = 1 if self.embeddings_classes[embedding_i] == self.embeddings_classes[embedding_j] else 0
                if pred == 1:
                    n_pos += 1
                else:
                    n_neg += 1
                self.embedding_pairs.append([
                    embedding_i,
                    embedding_j,
                    pred
                ])
        print(f"Number of positives: {n_pos}, negatives: {n_neg}")

    def pairs(self):
        for embedding_pair in self.embedding_pairs:
            yield (
                self.embeddings[embedding_pair[0]],
                self.embeddings[embedding_pair[1]],
                embedding_pair[2]
            )

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
