"""Implement Precision, Recall, F1-score, and accuracy from gold and predicated labels."""
# The input file contrain true_labels and pred_labels separated by a tab.
# It also verifies with the results from sklearn library
from argparse import ArgumentParser
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter


def compute_statistics_from_data(labels):
    """Compute Precision, Recall, and F1-score."""
    tp, fp, fn = 0, 0, 0
    label_wise_counts_dict = {}
    for label in labels:
        true, pred = label.split('\t')
        if true not in label_wise_counts_dict:
            label_wise_counts_dict[true] = {'TP': 0, 'FP': 0, 'FN': 0}
        if pred not in label_wise_counts_dict:
            label_wise_counts_dict[pred] = {'TP': 0, 'FP': 0, 'FN': 0}
        if pred == true:
            label_wise_counts_dict[true]['TP'] += 1
        else:
            label_wise_counts_dict[true]['FN'] += 1
            label_wise_counts_dict[pred]['FP'] += 1
    return label_wise_counts_dict


def compute_micro_average(label_wise_counts_dict):
    """Compute micro level Precision, Recall, and F1-score."""
    tp, fp, fn = 0, 0, 0
    for label, counts in label_wise_counts_dict.items():
        tp += counts['TP']
        fp += counts['FP']
        fn += counts['FN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def compute_macro_average(label_wise_counts_dict):
    """Compute macro level Precision, Recall, and F1-score."""
    precision, recall, f1_score = 0, 0, 0
    num_labels = len(label_wise_counts_dict)
    for label, counts in label_wise_counts_dict.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        precision_label = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_label = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_label = 2 * (precision_label * recall_label) / (precision_label + recall_label) if (precision_label + recall_label) > 0 else 0
        precision += precision_label
        recall += recall_label
        f1_score += f1_score_label
    # Average the metrics
    num_labels = len(label_wise_counts_dict)
    return precision / num_labels if num_labels > 0 else 0, recall / num_labels if num_labels > 0 else 0, f1_score / num_labels if num_labels > 0 else 0


def compute_weighted_average(label_wise_counts_dict, support):
    """Compute weighted level Precision, Recall, and F1-score."""
    precision, recall, f1_score = 0, 0, 0
    total_labels = sum(support.values())
    for label, counts in label_wise_counts_dict.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        # weight for each label
        weight = support[label] / total_labels
        precision_label = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_label = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_label = 2 * (precision_label * recall_label) / (precision_label + recall_label) if (precision_label + recall_label) > 0 else 0
        precision_label *= weight
        recall_label *= weight
        f1_score_label *= weight
        precision += precision_label
        recall += recall_label
        f1_score += f1_score_label
    # Weighted Average on all labels
    return precision, recall, f1_score


def read_lines_from_file(file_path):
    """Read lines from a file and return a list of stripped lines."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def main():
    """Main function to execute the script."""
    parser = ArgumentParser(description="Compute Precision, Recall, and F1-score.")
    parser.add_argument("--input", dest="inp", help="Path to the input file containing true and predicted labels.")
    args = parser.parse_args()
    labels = read_lines_from_file(args.inp)
    true_labels, pred_labels = list(zip(*[label.split('\t') for label in labels]))
    support = Counter(true_labels)
    label_wise_counts = compute_statistics_from_data(labels)
    precision, recall, f1 = compute_micro_average(label_wise_counts)
    print(f"Micro Precision: {precision:.4f}")
    print(f"Micro Recall: {recall:.4f}")
    print(f"Micro F1-score: {f1:.4f}")
    print("Using predefined libraries in sklearn")
    # Checking whether the code returns results equivalent to the sklearn library
    micro_precision_sklearn = precision_score(true_labels, pred_labels, average='micro')
    micro_recall_sklearn = recall_score(true_labels, pred_labels, average='micro')
    micro_f1_score_sklearn = f1_score(true_labels, pred_labels, average='micro')
    print(f"Micro Precision (sklearn): {micro_precision_sklearn:.4f}")
    print(f"Micro Recall (sklearn): {micro_recall_sklearn:.4f}")
    print(f"Micro F1-score (sklearn): {micro_f1_score_sklearn:.4f}")
    print('------------------')
    precision_macro, recall_macro, f1_macro = compute_macro_average(label_wise_counts)
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print("Using predefined libraries in sklearn")
    macro_precision_sklearn = precision_score(true_labels, pred_labels, average='macro')
    macro_recall_sklearn = recall_score(true_labels, pred_labels, average='macro')
    macro_f1_score_sklearn = f1_score(true_labels, pred_labels, average='macro')
    print(f"Macro Precision (sklearn): {macro_precision_sklearn:.4f}")
    print(f"Macro Recall (sklearn): {macro_recall_sklearn:.4f}")
    print(f"Macro F1-score (sklearn): {macro_f1_score_sklearn:.4f}")
    print('------------------')
    precision_weighted, recall_weighted, f1_weighted = compute_weighted_average(label_wise_counts, support)
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print(f"Weighted Recall: {recall_weighted:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")
    weighted_precision_sklearn = precision_score(true_labels, pred_labels, average='weighted')
    weighted_recall_sklearn = recall_score(true_labels, pred_labels, average='weighted')
    weighted_f1_score_sklearn = f1_score(true_labels, pred_labels, average='weighted')
    print("Using predefined libraries in sklearn")
    print(f"Weighted Precision (sklearn): {weighted_precision_sklearn:.4f}")
    print(f"Weighted Recall (sklearn): {weighted_recall_sklearn:.4f}")
    print(f"Weighted F1-score (sklearn): {weighted_f1_score_sklearn:.4f}")
    print('------------------')


if __name__ == "__main__":
    main()
