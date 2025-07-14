from matplotlib import pyplot as plt
from sklearn.metrics import auc

from analysis.stats_tools import pr_curve
from analysis.tmscore_dataset import TMscoreDataset

if __name__ == '__main__':

    THR = 0.8

    label='Structure Embeddings'
    dataloader = TMscoreDataset(
        tmscore_file="/home/joan/data/foldseek-rcsb/pdb-chain-ptm-scores.csv",
        thr=THR
    )
    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='red', linestyle='-', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    label='Foldseek'
    dataloader = TMscoreDataset(
        tmscore_file="/home/joan/data/foldseek-rcsb/pdb-chain-ptm-scores.csv",
        thr=THR,
        alt_scores_file="/home/joan/data/foldseek-rcsb/foldseek-exp.m8",
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[10])),
        reverse=False
    )

    recall, precision = pr_curve(dataloader)
    plt.plot(recall, precision, color='dodgerblue', linestyle='--', label=label)
    pr_auc = auc(recall, precision)
    print(f"AUC {label}", pr_auc)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.axis('square')

    plt.show()

