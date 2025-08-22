import argparse

from matplotlib import pyplot as plt
from numpy import linspace

from analysis.af_analysis_dataset import AfCathAnalysisDataset, Depth, cath_title
from qs_chain_bench import sensitivity_values

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

    depth = Depth.cath_archi

    dataloader = AfCathAnalysisDataset(
        score_file=structure_embeddings_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='red', linestyle='-', label='Structure Embeddings'
    )

    dataloader = AfCathAnalysisDataset(
        score_file=sequence_embeddings_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='orange', linestyle='-', label='Sequence Embeddings'
    )

    dataloader = AfCathAnalysisDataset(
        score_file=esm3_mean_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='burlywood', linestyle='-', label='ESM3 Mean'
    )

    dataloader = AfCathAnalysisDataset(
        score_file=tmvec_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='wheat', linestyle='--', label='TMvec'
    )

    dataloader = AfCathAnalysisDataset(
        score_file=foldseek_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[10])),
        dom_class_file=domain_class_file,
        score_reverse=False,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='dodgerblue', linestyle='--', label='Foldseek'
    )

    dataloader = AfCathAnalysisDataset(
        score_file=tmalign_scores,
        score_row_parser=lambda x: (x[0], x[1], float(x[2])),
        dom_class_file=domain_class_file,
        depth=depth
    )
    sen_values = sensitivity_values(dataloader)
    plt.plot(
        linspace(0, 1, len(sen_values)),
        [sen_values[i] for i in range(len(sen_values))],
        color='limegreen', linestyle='--', label='TMalign'
    )

    plt.xlabel('Fraction of Queries')
    plt.ylabel('Sensitivity')
    plt.title(f"CATH {cath_title(depth)}")
    plt.grid(True)
    plt.legend(loc='best')
    plt.axis('square')
    plt.savefig(f"{out_path}/qs-af-{cath_title(depth).lower()}-benchmark.png", bbox_inches='tight', dpi=300)
    plt.show()

