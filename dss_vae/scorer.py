from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from dss_vae.metrics.bleu import BleuScorer

scorer_dict = {
    'BLEU': BleuScorer.evaluate_file
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--score-cls", type=str, default='BLEU')
    parser.add_argument("--predict", "--p", dest="p", type=str)
    parser.add_argument("--reference", "--r", dest="r", type=str)
    parser.add_argument("--etype", type=str, default="corpus")

    args = parser.parse_args()
    print("Start Compute Score")
    score = scorer_dict[args.score_cls](args.p, args.r, score_type=args.etype)
    print(score)
