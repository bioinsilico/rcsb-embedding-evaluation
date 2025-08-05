import os
from operator import itemgetter

import pandas as pd
import numpy as np
from numpy import dot

import csv


class ScoreDataset:
    def __init__(
            self,
            embedding_path,
            query_file=None,
            target_file=None
    ):
        self.embedding_pairs = []
        self.embeddings = {}
        self.load_embedding(embedding_path)
        if query_file is not None and target_file is not None:
            self.compute_query_target_pairs(query_file, target_file)
        else:
            self.compute_embedding_pairs()

    def load_embedding(self, embedding_path):
        for e_i in os.listdir(embedding_path):
            v = np.array(list(pd.read_csv(f"{embedding_path}/{e_i}").iloc[:, 0].values))
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            self.embeddings[e_i.replace('.csv', '')] = v

    def compute_embedding_pairs(self):
        ids = list(self.embeddings.keys())
        while len(ids) > 0:
            e_i = ids.pop()
            for e_j in ids:
                self.__process_pair(e_i, e_j)
                self.__process_pair(e_j, e_i)
        self.__sort_pairs()

    def compute_query_target_pairs(self, query_file, target_file):
        print(f"Computing query-target pairs from {query_file} and {target_file}")
        query_ids = [r.strip() for r in open(query_file)]
        target_ids = [r.strip() for r in open(target_file)]
        for e_i in query_ids:
            for e_j in target_ids:
                self.__process_pair(e_i, e_j)
        self.__sort_pairs()

    def pairs(self):
        return self.embedding_pairs

    def __process_pair(self,e_i, e_j):
        if e_i not in self.embeddings or e_j not in self.embeddings:
            return
        self.embedding_pairs.append([
            e_i, e_j,
            dot(self.embeddings[e_i], self.embeddings[e_j]),

        ])

    def __sort_pairs(self):
        self.embedding_pairs = sorted(self.embedding_pairs, key=itemgetter(2))
        self.embedding_pairs.reverse()

def save_tsv(data, filepath, header=None, delimiter='\t', encoding='utf-8'):
    with open(filepath, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f, delimiter=delimiter)
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)