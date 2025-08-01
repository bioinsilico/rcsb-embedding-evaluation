import argparse

from matplotlib import pyplot as plt
from numpy import linspace
from analysis.tmscore_dataset import TMscoreDataset


def sensitivity_values(dataloader):
    sen_values = []
    for e_i in dataloader.domains():
        dom_scores = dataloader.get_domain_scores(e_i)
        fp_idx = [idx for idx, (e_j, s, tp, fp) in enumerate(dom_scores) if fp ==1]
        if len(fp_idx) > 0:
            fp_idx = fp_idx[0]
            sen = sum([tp for s, e_j, tp, fp in dom_scores[0: fp_idx]]) / dataloader.get_n_classes(e_i)
            sen_values.append(sen)
        else:
            sen_values.append(0)

    sen_values = sorted(sen_values)
    sen_values.reverse()
    return sen_values


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-chain-ptm-scores', type=str, required=True)
    parser.add_argument('--structure-embeddings-scores', type=str, required=True)
    parser.add_argument('--sequence-embeddings-scores', type=str, required=True)
    parser.add_argument('--tmvec-scores', type=str, required=True)
    parser.add_argument('--foldseek-scores', type=str, required=True)
    parser.add_argument('--esm3-mean-scores', type=str, required=True)
    parser.add_argument('--tmalign-scores', type=str, required=True)
    parser.add_argument('--score-threshold', type=float, required=True)
    parser.add_argument('--out-path', type=str, required=True)
    args = parser.parse_args()

    pdb_chain_ptm_scores = args.pdb_chain_ptm_scores
    structure_embeddings_scores = args.structure_embeddings_scores
    sequence_embeddings_scores = args.sequence_embeddings_scores
    tmvec_scores = args.tmvec_scores
    foldseek_scores = args.foldseek_scores
    esm3_mean_scores = args.esm3_mean_scores
    tmalign_scores = args.tmalign_scores
    score_threshold = args.score_threshold
    out_path = args.out_path

    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=structure_embeddings_scores,
        row_parser=lambda row: row
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='red', linestyle='-', label='Structure Embeddings'
    )

    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=sequence_embeddings_scores,
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[2])),
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='orange', linestyle='-', label='Sequence Embeddings'
    )

    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=esm3_mean_scores,
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[2])),
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='burlywood', linestyle='-', label='ESM3 Mean'
    )

    label='TMvec'
    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=tmvec_scores,
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[2])),
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='wheat', linestyle='--', label=label
    )

    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=foldseek_scores,
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[10])),
        reverse=False
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    dataloader = TMscoreDataset(
        ref_score_file=pdb_chain_ptm_scores,
        thr=score_threshold,
        alt_score_file=tmalign_scores,
        row_parser=lambda row: (row[0].replace("-","."), row[1].replace("-","."), float(row[2])),
    )
    sen_values = sensitivity_values(dataloader)
    sen_values.append(0)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='limegreen', linestyle='--', label='TMalign'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.title(f"TP TMscore > {score_threshold}")
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/qs-chain-lddt-{score_threshold}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()
