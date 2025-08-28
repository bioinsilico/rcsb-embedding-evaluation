import argparse

import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_histogram_from_tsv(file_path, output_file, column_index=0, bins=20):

    try:
        df = pd.read_csv(file_path, sep="\t", header=None)

        if column_index >= df.shape[1]:
            raise IndexError(f"Column index {column_index} out of range for file with {df.shape[1]} columns.")

        data = df.iloc[:, column_index]

        data = pd.to_numeric(data, errors="coerce").dropna()

        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins, edgecolor="black")
        plt.xlabel(f"Monomer TM-score")
        plt.ylabel("Assembly Pairs")
        plt.title(f"10,000 FP Assembly Pairs")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fp-score-list.tsv', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    file_path = args.fp_score_list_tsv
    output_file= args.output_file

    plot_histogram_from_tsv(file_path, output_file)