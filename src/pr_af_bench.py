import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import auc

from analysis.af_analysis_dataset import AfCathAnalysisDataset
from analysis.stats_tools import pr_curve

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--domain-class-file', type=str, required=True)
    parser.add_argument('--structure-embeddings-scores', type=str, required=True)
    parser.add_argument('--sequence-embeddings-scores', type=str, required=True)
    parser.add_argument('--tmvec-scores', type=str, required=True)
    parser.add_argument('--foldseek-scores', type=str, required=True)
    parser.add_argument('--esm3-mean-scores', type=str, required=True)
    parser.add_argument('--tmalign-scores', type=str, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    domain_class_file = args.domain_class_file
    structure_embeddings_scores = args.structure_embeddings_scores
    sequence_embeddings_scores = args.sequence_embeddings_scores
    tmvec_scores = args.tmvec_scores
    foldseek_scores = args.foldseek_scores
    esm3_mean_scores = args.esm3_mean_scores
    tmalign_scores = args.tmalign_scores
    out_path = args.out_path

    label='Structure Embeddings'
    dataloader = AfCathAnalysisDataset(
        score_file=structure_embeddings_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='red', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'Sequence Embeddings'
    dataloader = AfCathAnalysisDataset(
        score_file=sequence_embeddings_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='orange', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'ESM3 Mean'
    dataloader = AfCathAnalysisDataset(
        score_file=esm3_mean_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='burlywood', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'TMvec'
    dataloader = AfCathAnalysisDataset(
        score_file=tmvec_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='wheat', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'Foldseek'
    dataloader = AfCathAnalysisDataset(
        score_file=foldseek_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[11])),
        dom_class_file=domain_class_file,
        score_reverse=True
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='dodgerblue', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'TMalign'
    dataloader = AfCathAnalysisDataset(
        score_file=tmalign_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='limegreen', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"CATH Topology")
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/pr-af-topology-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()

