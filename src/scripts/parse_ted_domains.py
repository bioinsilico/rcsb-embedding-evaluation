import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--excluded-proteins-file', type=str, required=True)
    parser.add_argument('--ted_domains_file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    excluded_proteins = set()
    for _id in open(args.excluded_proteins_file, 'r'):
        excluded_proteins.add(_id.strip())

    with open(args.output_file, 'w') as f:
        for line in open(args.ted_domains_file, 'r'):
            row = line.strip().split("\t")
            _id = row[0]
            if _id not in excluded_proteins:
                f.write(f"{_id}\t{row[1]}\n")