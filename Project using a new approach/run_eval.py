import argparse
from pprint import pprint

from src.eval_metrics import evaluate_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_labels", required=True, help="Path to train_labels.csv")
    p.add_argument("--submission", required=True, help="Path to submission.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scores = evaluate_all(args.train_labels, args.submission)
    pprint(scores)

