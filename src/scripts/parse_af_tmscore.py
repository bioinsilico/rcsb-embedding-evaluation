import argparse
import csv
import re

def parse_tmscore_file(file_path):
    """
    Parses a file with TM-score data, extracting the maximum TM-score
    and associated filenames for each pair of lines.

    Args:
        file_path (str): The path to the input file.

    Returns:
        list: A list of tuples, where each tuple contains the two filenames
              and the maximum TM-score for that pair.
    """
    results = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            line1 = lines[i].strip()
            line2 = lines[i+1].strip()

            pattern = r'>/AF-(.*?)\.cif:.*TM-score=([0-9.]+)'

            match1 = re.search(pattern, line1)
            match2 = re.search(pattern, line2)

            if match1 and match2:
                filename1 = f'AF-{match1.group(1)}'
                tmscore1 = float(match1.group(2))

                filename2 = f'AF-{match2.group(1)}'
                tmscore2 = float(match2.group(2))

                max_tmscore = max(tmscore1, tmscore2)
                results.append((filename1, filename2, max_tmscore))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm-score-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()
    parsed_data = parse_tmscore_file(args.tm_score_file)
    with open(args.output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        tsv_writer = csv.writer(tsvfile, delimiter='\t')
        tsv_writer.writerows(parsed_data)
