
import numpy as np
from numpy import dot
from operator import itemgetter
from scipy.stats import spearmanr, pearsonr, kendalltau
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.analysis_dataset import AnalysisDataset, is_tp
from analysis.correlation_dataset import CorrelationDataset


def fold_fp(c, c_anchor):
    c_tree = c.split(".")
    c_anchor_tree = c_anchor.split(".")
    if c_tree[0] == c_anchor_tree[0] and c_tree[1] == c_anchor_tree[1]:
        return False
    return True


def get_sensitivity_query_fraction(
        embedding_path,
        embedding_class_file,
        depth
):
    dataloader = AnalysisDataset(
        embedding_path,
        embedding_class_file,
        depth
    )

    sen_values = []
    for dom_i, embedding_i in dataloader.domains():
        class_i = dataloader.get_class(dom_i)
        score = [
            (dom_j, dataloader.get_class(dom_j), dot(embedding_i, embedding_j))
            for dom_j, embedding_j in dataloader.domains(n=0)
        ]
        sort_score = sorted([(d, c, '%.2f' % s) for (d, c, s) in score if dom_i != d], key=itemgetter(2))
        sort_score.reverse()
        fp_idx = [idx for (idx, (d, c, s)) in enumerate(sort_score) if fold_fp(c, class_i)][0]
        tp = [(d, c, s) for (d, c, s) in sort_score[0:fp_idx] if is_tp(depth, class_i, c)]
        n_classes = dataloader.get_n_classes(dom_i)
        sen = len(tp) / n_classes
        sen_values.append(sen)

    sen_values = sorted(sen_values)
    sen_values.reverse()
    return sen_values


def pr_curve(dataloader):
    n_tp = 0
    n_fp = 0
    recall = [0]
    precision = [1]

    with tqdm(total=dataloader.pairs_len(), desc="Domain Pairs", unit="pair") as pbar:
        n_pos = dataloader.get_n_pos()
        for idx, (tp, fp) in enumerate(dataloader.pairs()):
            if idx > 0 and idx % 1000 == 0 and (n_tp+n_fp) > 0:
                recall.append(n_tp / n_pos)
                precision.append(n_tp / (n_tp+n_fp))
            n_fp += fp
            n_tp += tp
            pbar.update(1)

    recall.append(n_tp / n_pos)
    precision.append(n_tp / (n_tp+n_fp))

    return recall, precision


def plot_2d_points(points, title="2D Point Plot", xlabel="TM-score", ylabel="pTM-score",
                   color='blue', marker='o', grid=True):
    """
    Plots a collection of 2D points.

    Parameters:
        points (list of tuple): List of 2D points [(x1, y1), (x2, y2), ...].
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        color (str): Color of the points.
        marker (str): Marker style for the points.
        grid (bool): Whether to show a grid on the plot.
    """
    # Unpack points into X and Y components
    print("Plotting 2D points")
    x_coords, y_coords = zip(*points)

    # Create the plot
    plt.figure(figsize=(8, 8))

    plt.scatter(x_coords, y_coords, color=color, marker=marker)

    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Optional grid
    if grid:
        plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('square')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("2d-points.png")
    # Show the plot
    plt.show()


def discrete_score(s):
    if s >= 0.6:
        return 1
    return 0


def get_tm_score_correlation(
        embedding_path,
        tm_score_file
):
    dataloader = CorrelationDataset(
        embedding_path,
        tm_score_file
    )

    corrs = []
    for d_i, e_i in dataloader.domains():
        scores = [
            (discrete_score(s), discrete_score(dot(e_i, e_j)))
            for d_j, e_j, s in dataloader.get_domain_scores(d_i)
        ]
        if len(scores) == 0:
            continue
        # if if_with_probability(0.005):
        #    plot_2d_points(scores)
        corr_p, p_value = pearsonr(
            np.array([pair[0] for pair in scores]),
            np.array([pair[1] for pair in scores]),
        )
        corrs.append(corr_p)

    return corrs
