from matplotlib import pyplot as plt
from numpy import linspace
from analysis.tmscore_dataset import TMscoreDataset

if __name__ == '__main__':

    THR = 0.8

    dataloader = TMscoreDataset(
        tmscore_file="/home/joan/data/foldseek-rcsb/pdb-chain-ptm-scores.csv",
        thr=THR
    )

    sen_values = []
    for e_i in dataloader.domains():
        dom_scores = dataloader.get_domain_scores(e_i)
        fp_idx = [idx for idx, (e_j, s, tp, fp) in enumerate(dom_scores) if fp ==1][0]
        sen = sum([tp for s, e_j, tp, fp in dom_scores[0: fp_idx]]) / dataloader.get_n_classes(e_i)
        sen_values.append(sen)

    sen_values = sorted(sen_values)
    sen_values.reverse()

    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='red', linestyle='-', label='Structure Embeddings'
    )

    dataloader = TMscoreDataset(
        tmscore_file="/home/joan/data/foldseek-rcsb/pdb-chain-ptm-scores.csv",
        thr=THR,
        alt_scores_file="/home/joan/data/foldseek-rcsb/foldseek-exp.m8",
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[10])),
        reverse=False
    )

    sen_values = []
    for e_i in dataloader.domains():
        dom_scores = dataloader.get_domain_scores(e_i)
        fp_idx = [idx for idx, (e_j, s, tp, fp) in enumerate(dom_scores) if fp ==1][0]
        sen = sum([tp for s, e_j, tp, fp in dom_scores[0: fp_idx]]) / dataloader.get_n_classes(e_i)
        sen_values.append(sen)

    sen_values = sorted(sen_values)
    sen_values.reverse()

    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.show()

