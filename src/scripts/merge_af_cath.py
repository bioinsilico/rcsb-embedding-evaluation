import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cath-file', type=str, required=True)
    parser.add_argument('--af-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    up_ids = set()
    cath_map = {}

    for line in open(args.af_file, "r"):
        up_ids.add(line.strip())

    for line in open(args.cath_file, "r"):
        up_acc, cath_id = line.strip().split("\t")
        if up_acc not in up_ids:
            continue
        if up_acc not in cath_map:
            cath_map[up_acc] = set()
        cath_map[up_acc].add(cath_id)

    with open(args.output_file, "w") as out_file:
        for up_acc in up_ids:
            if up_acc in cath_map:
                cath_ids = list(cath_map[up_acc])
                cath_ids.sort()
                out_file.write(f"{up_acc}\t{','.join(cath_ids)}\n")