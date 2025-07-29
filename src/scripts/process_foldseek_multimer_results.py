import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldseek-results', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    scores = {}
    for line in open(args.foldseek_results, "r"):
        row = line.split("\t")
        score = float(row[5])
        if row[0] not in scores:
            scores[row[0]] = {}
        if row[1] not in scores[row[0]]:
            scores[row[0]][row[1]] = 0
        if score > scores[row[0]][row[1]]:
            scores[row[0]][row[1]] = score

    with open(args.output_file, "w") as out_file:
        for a_i in scores:
            for a_j in scores[a_i]:
                out_file.write(f"{a_i}\t{a_j}\t{scores[a_i][a_j]}\n")