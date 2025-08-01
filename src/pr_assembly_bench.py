import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import auc

from analysis.stats_tools import pr_curve
from analysis.tmscore_dataset import TMscoreDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-chain-ptm-scores', type=str, required=True)
    parser.add_argument('--structure-embeddings-scores', type=str, required=True)
    parser.add_argument('--sequence-embeddings-scores', type=str, required=True)
    parser.add_argument('--esm3-mean-scores', type=str, required=True)
    parser.add_argument('--foldseek-scores', type=str, required=True)
    parser.add_argument('--tmalign-scores', type=str, required=True)
    parser.add_argument('--tmscore-threshold', type=float, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    pdb_chain_ptm_scores = args.pdb_chain_ptm_scores
    structure_embeddings_scores = args.structure_embeddings_scores
    sequence_embeddings_scores = args.sequence_embeddings_scores
    esm3_mean_scores = args.esm3_mean_scores
    foldseek_scores = args.foldseek_scores
    tmalign_scores = args.tmalign_scores
    tmscore_threshold = args.tmscore_threshold
    out_path = args.out_path

    label='Structure Embeddings'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        alt_score_file=structure_embeddings_scores,
        thr=tmscore_threshold,
        row_parser=lambda row: (row[0], row[1], float(row[2]))
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='red', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='Sequence Embeddings'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=tmscore_threshold,
        alt_score_file=sequence_embeddings_scores,
        row_parser=lambda row: (row[0], row[1], float(row[2]))
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='orange', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='ESM3 Mean'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=tmscore_threshold,
        alt_score_file=esm3_mean_scores,
        row_parser=lambda row: (row[0], row[1], float(row[2]))
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='burlywood', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='Foldseek'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=tmscore_threshold,
        alt_score_file=foldseek_scores,
        row_parser=lambda row: (row[0], row[1], float(row[2]))
    )

    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='dodgerblue', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label = 'TMalign'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=tmscore_threshold,
        alt_score_file=tmalign_scores,
        row_parser=lambda row: (row[0], row[1], float(row[2]))
    )
    recall, precision = pr_curve(dataloader)
    recall.append(1)
    precision.append(0)
    plt.plot(recall, precision, color='limegreen', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"TP TMscore > {tmscore_threshold}")
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/pr-assembly-{tmscore_threshold}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()

