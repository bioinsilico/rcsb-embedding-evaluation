import argparse

from numpy import linspace
import matplotlib.pyplot as plt

from analysis.analysis_dataset import Depth, is_tp
from analysis.stats_tools import get_sensitivity_query_fraction, fold_fp
from utils.extract_foldseek_scores import process_foldseek_data



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-embeddings', type=str, required=True)
    parser.add_argument('--sequence-embeddings', type=str, required=True)
    parser.add_argument('--mean-embeddings', type=str, required=True)
    parser.add_argument('--results-path', type=str, required=True)
    parser.add_argument('--domain-classes', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    structure_embedding_folder = args.structure_embeddings
    sequence_embedding_folder = args.sequence_embeddings
    mean_embedding_folder = args.mean_embeddings
    results_path = args.results_path
    domain_class_file = args.domain_classes
    out_path = args.out_path

    depth = Depth.scop_family

    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1.1)

    values = get_sensitivity_query_fraction(
        sequence_embedding_folder,
        domain_class_file,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='orange', linestyle='-', label='Sequence Embeddings'
    )

    values = get_sensitivity_query_fraction(
        mean_embedding_folder,
        domain_class_file,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='orange', linestyle='-', label='Mean ESM3 Embeddings'
    )

    values = get_sensitivity_query_fraction(
        structure_embedding_folder,
        domain_class_file,
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='red', linestyle='-', label='Structure Embeddings'
    )

    values = process_foldseek_data(
        f'{results_path}/foldseek.txt',
        domain_class_file,
        lambda row: (".".join(row[0].split(".")[:-1]), ".".join(row[1].split(".")[:-1]), int(row[11])),
        depth,
        is_tp,
        fold_fp
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    values = process_foldseek_data(
        f'{results_path}/TMalign.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth,
        is_tp,
        fold_fp
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='limegreen', linestyle='--', label='TMalign'
    )

    values = process_foldseek_data(
        f'{results_path}/dali.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth,
        is_tp,
        fold_fp
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='mediumorchid', linestyle='--', label='Dali'
    )

    values = process_foldseek_data(
        f'{results_path}/pdb-foldseek-zer.txt',
        domain_class_file,
        lambda row: (row[0], row[1], float(row[2])),
        depth,
        is_tp,
        fold_fp
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='yellowgreen', linestyle='--', label='BioZernike'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.title('SCOPe Family')
    plt.grid(True)
    # plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/foldseek-{depth}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()
