import argparse

from numpy import linspace
import matplotlib.pyplot as plt

from analysis.analysis_dataset import Depth, depth_name
from analysis.stats_tools import get_sensitivity_query_fraction
from analysis.query_sensitivity import qs_scores


def plot_sensitivity(
    structure_embedding_folder,
    foldsee_results,
    domain_class_file,
    out_path,
    depth,
    out_tag
):
    plt.figure(figsize=(8, 6))
    plt.ylim(0, 1.1)

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

    values = qs_scores(
        foldsee_results,
        domain_class_file,
        lambda row: (row[0], row[1], float(row[11])),
        depth
    )
    plt.plot(
        linspace(0, 1, len(values)),
        [values[i] for i in range(len(values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.title(depth_name(depth))
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/qs-{depth}-exc-{out_tag}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-embeddings', type=str, required=True)
    parser.add_argument('--foldseek-results', type=str, required=True)
    parser.add_argument('--domain-classes', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--out-tag', type=str, required=True)
    args = parser.parse_args()

    structure_embedding_folder = args.structure_embeddings
    foldseek_results = args.foldseek_results
    domain_class_file = args.domain_classes
    out_path = args.out_path
    out_tag = args.out_tag

    plot_sensitivity(
        structure_embedding_folder,
        foldseek_results,
        domain_class_file,
        out_path,
        Depth.scop_family,
        out_tag
    )

    plot_sensitivity(
        structure_embedding_folder,
        foldseek_results,
        domain_class_file,
        out_path,
        Depth.scop_super_family,
        out_tag
    )

    plot_sensitivity(
        structure_embedding_folder,
        foldseek_results,
        domain_class_file,
        out_path,
        Depth.scop_fold,
        out_tag
    )
