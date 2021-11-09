from nltk.translate import bleu_score

from .base_metric import BaseMetric


def top_selection(reference, hypothesis):
    new_reference = []
    for refs, hypo in zip(reference, hypothesis):
        bleu_scores = [bleu_score.sentence_bleu(ref, hypo) for ref in refs]
        new_reference.append(refs[bleu_scores.index(max(bleu_scores))])
    return new_reference


class BleuScoreMetric(BaseMetric):
    def __init__(self):
        super(BleuScoreMetric, self).__init__(name="BLEU")

    def _evaluating(self, reference, hypothesis, etype="corpus"):
        self._check_format(reference, hypothesis)
        eval_select = {
            'corpus': bleu_score.corpus_bleu,
            'sents': bleu_score.sentence_bleu,
        }[etype]
        return 100.0 * eval_select(list_of_references=reference, hypotheses=hypothesis)

    def _check_format(self, reference, hypothesis):
        if not isinstance(reference, list):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0], list):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0][0], list):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))
        elif not isinstance(reference[0][0][0], str):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))

        if not isinstance(hypothesis, list):
            raise RuntimeError(
                "hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))
        elif not isinstance(hypothesis[0], list):
            raise RuntimeError(
                "hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))
        elif not isinstance(hypothesis[0][0], str):
            raise RuntimeError(
                "hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))

    @staticmethod
    def evaluate_file(pred_file, gold_files, etype="corpus"):

        with open(pred_file, "r") as f:
            hypothesis = [line.strip().split(" ") for line in f.readlines()]

        references = [[] for _ in range(len(hypothesis))]
        if not isinstance(gold_files, list):
            gold_files = [gold_files, ]

        for file in gold_files:
            with open(file, "r") as f:
                ref_i = [line.strip().split(" ") for line in f.readlines()]
                for idx, ref in enumerate(ref_i):
                    references[idx].append(ref)

        evaluator = BleuScoreMetric()
        bleu = evaluator._evaluating(reference=references, hypothesis=hypothesis, etype=etype)
        return bleu
