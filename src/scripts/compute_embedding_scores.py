import argparse

from analysis.score_dataset import ScoreDataset, save_tsv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings-path', type=str, required=True)
    parser.add_argument('--score-file', type=str, required=True)
    parser.add_argument('--query-file', type=str)
    parser.add_argument('--target-file', type=str)
    args = parser.parse_args()

    structure_embedding_folder = args.embeddings_path
    score_file = args.score_file

    dataloader = ScoreDataset(
        embedding_path=structure_embedding_folder,
        query_file=args.query_file,
        target_file=args.target_file
    )

    save_tsv(dataloader.pairs(), score_file)