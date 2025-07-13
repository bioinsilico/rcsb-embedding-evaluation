import argparse
import csv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--score-file', type=str, required=True)
    parser.add_argument('--domain-class-file', type=str, required=True)
    parser.add_argument('--results-file', type=str, required=True)
    args = parser.parse_args()

    score_file = args.score_file
    domain_class_file = args.domain_class_file
    results_file = args.results_file
    out_path = args.out_path
    out_tag = args.out_tag

    filtered = set()
    with open(domain_class_file, "r") as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_reader:
            filtered.add(row[0])

    with open(score_file, "r") as tsv_file:
        with open(results_file, "w") as res_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            for row in tsv_reader:
                row[0] = row[0].replace(".pdb", "")
                row[1] = row[1].replace(".pdb", "")
                if row[0] in filtered and row[1] in filtered:
                    res_file.write("\t".join(row)+"\n")
