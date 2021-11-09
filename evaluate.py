import argparse
import os

import nltk
import numpy as np
import rouge
# import sys
from nltk.translate.bleu_score import corpus_bleu
from pythonrouge.pythonrouge import Pythonrouge

# sys.path.append('/home/liuxg/workspace/SAparaphrase/')
# sys.path.append('/home/liuxg/workspace/SAparaphrase/bert')
# from rouge import Rouge
bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


class Option(object):
    def __init__(self, d):
        self.__dict__ = d

    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))


def get_corpus_bleu_scores(actual_word_lists, generated_word_lists):
    bleu_scores = dict()
    for i in range(len(bleu_score_weights)):
        bleu_scores[i + 1] = round(
            corpus_bleu(
                list_of_references=actual_word_lists[:len(generated_word_lists)],
                hypotheses=generated_word_lists,
                weights=bleu_score_weights[i + 1]), 4)

    return bleu_scores


def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='bleu', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--generated_path', default=None, type=str)

    d = vars(parser.parse_args())
    option = Option(d)
    if option.mode == 'bleu':
        evaluate_bleu(option.reference_path, option.generated_path)
    elif option.mode == 'bleu_c':
        evaluate_bleu_corpus(option.reference_path, option.generated_path)
    elif option.mode == 'bleu2':
        evaluate_bleu2(option.reference_path, option.generated_path)
    # elif option.mode == 'semantic':
    #     evaluate_semantic(option.reference_path, option.generated_path)
    elif option.mode == 'rouge':
        evaluate_rouge(option.reference_path, option.generated_path, False)
    elif option.mode == 'multi-rouge':
        evaluate_rouge(option.reference_path, option.generated_path)
    else:
        print('not a valid argument')


def evaluate_bleu_corpus(reference_path, generated_path):
    # Evaluate model scores
    actual_word_lists = []
    with open(reference_path) as f:
        for line in f:
            if '#' in line:
                sents = line.strip().lower().split('#')
                actual_word_lists.append([x.split() for x in sents])
            else:
                actual_word_lists.append([line.strip().lower().split()])

    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip().lower().split())
    actual_word_lists = actual_word_lists[:len(generated_word_lists)]
    bleu_scores = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)
    sumss = 0
    for s in bleu_scores:
        sumss += 0.25 * bleu_scores[s]
    print('bleu scores:', sumss, bleu_scores)


def evaluate_bleu(reference_path, generated_path):
    # Evaluate model scores
    actual_word_lists = []
    with open(reference_path) as f:
        for line in f:
            if '#' in line:
                sents = line.strip().lower().split('#')
                actual_word_lists.append([x.split() for x in sents])
            else:
                actual_word_lists.append([line.strip().lower().split()])

    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip().lower().split())
    actual_word_lists = actual_word_lists[:len(generated_word_lists)]

    bleu_scores = [nltk.translate.bleu_score.sentence_bleu(
        a, g, smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
        for a, g in zip(actual_word_lists, generated_word_lists)]
    print('sentence level bleu', np.mean(bleu_scores))


def evaluate_bleu2(reference_path, generated_path):
    # Evaluate model scores
    actual_word_lists = []
    with open(reference_path) as f:
        for line in f:
            if '#' in line:
                sents = line.strip().lower().split('#')
                actual_word_lists.append([x.split() for x in sents])
            else:
                actual_word_lists.append([line.strip().lower().split()])

    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip().lower().split())
    actual_word_lists = actual_word_lists[:len(generated_word_lists)]

    bleu_scores = [nltk.translate.bleu_score.sentence_bleu(
        a, g, weights=[0.5, 0.5, 0, 0],
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1)
        for a, g in zip(actual_word_lists, generated_word_lists)]
    print('sentence level bleu', np.mean(bleu_scores))


# def evaluate_semantic(reference_path, generated_path):
#     actual_word_lists = []
#     with open(reference_path) as f:
#         for line in f:
#             actual_word_lists.append(line.strip())
#
#     generated_word_lists = []
#     with open(generated_path) as f:
#         for line in f:
#             generated_word_lists.append(line.strip())
#
#     model = BertEncoding()
#     rep1s, rep2s = [], []
#     batchsize = int(100)
#     for i in range(int(len(generated_word_lists) / batchsize)):
#         s1s = actual_word_lists[i * batchsize:i * batchsize + batchsize]
#         s2s = generated_word_lists[i * batchsize:i * batchsize + batchsize]
#         rep1 = model.get_encoding(s1s)
#         rep1s.append(rep1)
#         rep2 = model.get_encoding(s2s)
#         rep2s.append(rep2)
#     rep1 = torch.cat(rep1s, 0)
#     rep2 = torch.cat(rep2s, 0)
#     summation = torch.sum(rep1 * rep2, 1) / (rep1.norm(2, 1) * rep2.norm(2, 1))
#     print(torch.mean(summation))


def evaluate_rouge(reference_path, generated_path, multi=True):
    # Evaluate model scores
    actual_word_lists = []
    multi_flag = True
    with open(reference_path) as f:
        for line in f:
            if multi:
                sents = line.strip().lower().split('#')
                actual_word_lists.append([x for x in sents])
            else:
                actual_word_lists.append([line.strip().lower()])
    generated_word_lists = []
    with open(generated_path) as f:
        for line in f:
            generated_word_lists.append(line.strip().lower())
    actual_word_lists = actual_word_lists[:len(generated_word_lists)]
    print(actual_word_lists[0], len(generated_word_lists))

    for aggregator in ['Avg', 'Best', 'Individual']:
        print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=4,
                                limit_length=True,
                                length_limit=100,
                                length_limit_type='words',
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=0.5,  # Default F1_score
                                weight_factor=1.2,
                                stemming=True)
        all_hypothesis = generated_word_lists
        all_references = actual_word_lists
        scores = evaluator.get_scores(all_hypothesis, all_references)

        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best:
                # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(metric, results_per_ref['p'][reference_id],
                                                     results_per_ref['r'][reference_id],
                                                     results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(metric, results['p'], results['r'], results['f']))

        rouge_score = Pythonrouge(
            summary_file_exist=False,
            summary=generated_word_lists, reference=actual_word_lists,
            n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
            recall_only=True, stemming=True, stopwords=True,
            word_level=True, length_limit=True, length=50,
            use_cf=False, cf=95, scoring_formula='average',
            resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge_score.calc_score()
        print(score)


def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


# def test_semantic(s1, s2):
#     model = BertSimilarity()
#     rep1 = model.get_encoding(s1, s1)
#     rep2 = model.get_encoding(s1, s2)
#     rep3 = model.get_encoding(s2, s2)
#     rep1 = (rep1 + rep3) / 2
#     semantic = torch.sum(rep1 * rep2, 1) / (rep1.norm() * rep2.norm())
#     semantic = semantic * (1 - (abs(rep1.norm() - rep2.norm()) / max(rep1.norm(), rep2.norm())))
#     print(torch.mean(semantic))


if __name__ == "__main__":
    main()
