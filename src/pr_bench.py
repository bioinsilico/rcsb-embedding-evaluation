import argparse

from sklearn.metrics import auc
import matplotlib.pyplot as plt

from analysis.analysis_dataset import AnalysisDataset, Depth, depth_name
from analysis.stats_tools import pr_curve


def plot_pr(
    structure_embeddings_score ,
    sequence_embeddings_score,
    mean_embeddings_score ,
    domain_class_file,
    results_path,
    out_path,
    depth,
    legend=False
):
    label='Structure Embeddings'
    dataloader = AnalysisDataset(
        score_file=structure_embeddings_score,
        score_row_parser=lambda row: row,
        dom_class_file=domain_class_file,
        depth=depth
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='red', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    """label='Sequence Embeddings'
    dataloader.reload_embedding_pairs(
        score_file=sequence_embeddings_score,
        score_row_parser=lambda row: row
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='orange', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='ESM3 Mean'
    dataloader.reload_embedding_pairs(
        score_file=mean_embeddings_score,
        score_row_parser=lambda row: row
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='burlywood', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)"""

    label='Foldseek'
    dataloader.reload_embedding_pairs(
        score_file=f"{results_path}/foldseek.txt",
        score_row_parser=lambda row: (row[0][0:7], row[1][0:7], float(row[10])),
        score_reverse=True
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='dodgerblue', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='TMalign'
    dataloader.reload_embedding_pairs(
        score_file=f"{results_path}/TMalign.txt",
        score_row_parser=lambda row: (row[0], row[1], float(row[2])),
        score_reverse=False
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='limegreen', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='Dali'
    dataloader.reload_embedding_pairs(
        score_file=f"{results_path}/dali.txt",
        score_row_parser=lambda row: (row[0], row[1], float(row[2])),
        score_reverse=False
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='mediumorchid', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    """label='TMvec'
    dataloader.reload_embedding_pairs(
        score_file=f"{results_path}/tmvec.txt",
        score_row_parser=lambda row: (row[0], row[1], float(row[2])),
        score_reverse=False
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='wheat', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='BioZernike'
    dataloader.reload_embedding_pairs(
        score_file=f"{results_path}/pdb-foldseek-zer.txt",
        score_row_parser=lambda row: (row[0], row[1], float(row[2])),
        score_reverse=False
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='yellowgreen', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)"""

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(depth_name(depth))
    plt.grid(True)
    if legend:
        plt.legend(loc='best')
    plt.axis('square')
    #plt.savefig(f"{out_path}/pr-{depth}-benchmark.png", bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-embeddings-score', type=str, required=True)
    parser.add_argument('--sequence-embeddings-score', type=str, required=True)
    parser.add_argument('--mean-embeddings-score', type=str, required=True)
    parser.add_argument('--domain-class-file', type=str, required=True)
    parser.add_argument('--results-path', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    structure_embeddings_score = args.structure_embeddings_score
    sequence_embeddings_score = args.sequence_embeddings_score
    mean_embeddings_score = args.mean_embeddings_score
    domain_class_file = args.domain_class_file
    results_path = args.results_path
    out_path = args.out_path

    plot_pr(
        structure_embeddings_score ,
        sequence_embeddings_score,
        mean_embeddings_score ,
        domain_class_file,
        results_path,
        out_path,
        Depth.scop_family,
        legend=False
    )

    """plot_pr(
        structure_embeddings_score ,
        sequence_embeddings_score,
        mean_embeddings_score ,
        domain_class_file,
        results_path,
        out_path,
        Depth.scop_super_family,
        legend=False
    )

    plot_pr(
        structure_embeddings_score ,
        sequence_embeddings_score,
        mean_embeddings_score ,
        domain_class_file,
        results_path,
        out_path,
        Depth.scop_fold,
        legend=True
    )"""
