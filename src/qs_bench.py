import argparse

from numpy import linspace
import matplotlib.pyplot as plt

from analysis.analysis_dataset import Depth, depth_name
from analysis.query_sensitivity import qs_scores

def plot_sensitivity(
    structure_embeddings_score,
    sequence_embedding_score,
    mean_embedding_score,
    results_path,
    domain_class_file,
    out_path,
    depth,
    legend=False
):
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1.1)

    values = qs_scores(
        structure_embeddings_score,
        domain_class_file,
        lambda row: row,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='red', linestyle='-', label='Structure Embeddings'
    )

    values = qs_scores(
        sequence_embedding_score,
        domain_class_file,
        lambda row: row,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='orange', linestyle='-', label='Sequence Embeddings'
    )

    values = qs_scores(
        mean_embedding_score,
        domain_class_file,
        lambda row: row,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='burlywood', linestyle='-', label='ESM3 Mean'
    )

    values = qs_scores(
        f'{results_path}/foldseek.txt',
        domain_class_file,
        lambda row: (".".join(row[0].split(".")[:-1]), ".".join(row[1].split(".")[:-1]), int(row[11])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    values = qs_scores(
        f'{results_path}/TMalign.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='limegreen', linestyle='--', label='US-align'
    )

    values = qs_scores(
        f'{results_path}/dali.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='mediumorchid', linestyle='--', label='Dali'
    )

    values = qs_scores(
        f'{results_path}/tmvec.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='wheat', linestyle='--', label='TMvec'
    )

    values = qs_scores(
        f'{results_path}/pdb-foldseek-zer.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='yellowgreen', linestyle='--', label='BioZernike'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.title(depth_name(depth))
    plt.grid(True)
    if legend:
        plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/qs-{depth}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-embeddings-score', type=str, required=True)
    parser.add_argument('--sequence-embeddings-score', type=str, required=True)
    parser.add_argument('--mean-embeddings-score', type=str, required=True)
    parser.add_argument('--results-path', type=str, required=True)
    parser.add_argument('--domain-classes', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    structure_embeddings_score = args.structure_embeddings_score
    sequence_embedding_score = args.sequence_embeddings_score
    mean_embedding_score = args.mean_embeddings_score
    results_path = args.results_path
    domain_class_file = args.domain_classes
    out_path = args.out_path

    plot_sensitivity(
        structure_embeddings_score,
        sequence_embedding_score,
        mean_embedding_score,
        results_path,
        domain_class_file,
        out_path,
        Depth.scop_family,
        legend=True
    )

    plot_sensitivity(
        structure_embeddings_score,
        sequence_embedding_score,
        mean_embedding_score,
        results_path,
        domain_class_file,
        out_path,
        Depth.scop_super_family,
        legend=False
    )

    plot_sensitivity(
        structure_embeddings_score,
        sequence_embedding_score,
        mean_embedding_score,
        results_path,
        domain_class_file,
        out_path,
        Depth.scop_fold,
        legend=False
    )
