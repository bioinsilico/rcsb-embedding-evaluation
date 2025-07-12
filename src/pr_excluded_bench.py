import argparse

from sklearn.metrics import auc
import matplotlib.pyplot as plt

from analysis.analysis_dataset import AnalysisDataset, Depth, depth_name
from analysis.stats_tools import pr_curve


def plot_pr(
    structure_embeddings_score ,
    domain_class_file,
    foldseek_result,
    out_path,
    depth,
    out_tag,
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

    label='Foldseek'
    dataloader.reload_embedding_pairs(
        score_file=foldseek_result,
        score_row_parser=lambda row: (row[0][0:7], row[1][0:7], float(row[10])),
        score_reverse=True
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='dodgerblue', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(depth_name(depth))
    plt.grid(True)
    if legend:
        plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/pr-{depth}-exc-{out_tag}-benchmark.png", bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--structure-embeddings-score', type=str, required=True)
    parser.add_argument('--domain-class-file', type=str, required=True)
    parser.add_argument('--foldseek-result', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--out-tag', type=str, required=True)
    args = parser.parse_args()

    structure_embeddings_score = args.structure_embeddings_score
    domain_class_file = args.domain_class_file
    foldseek_result = args.foldseek_result
    out_path = args.out_path
    out_tag = args.out_tag

    plot_pr(
        structure_embeddings_score ,
        domain_class_file,
        foldseek_result,
        out_path,
        Depth.scop_family,
        out_tag,
        legend=False
    )

