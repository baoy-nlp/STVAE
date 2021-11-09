from nltk.translate import meteor_score

from .base_metric import BaseMetric


class MeteorScoreMetric(BaseMetric):
    def __init__(self):
        super(MeteorScoreMetric, self).__init__(name="BLEU")

    def _evaluating(self, reference, hypothesis, etype="corpus"):
        self._check_format(reference, hypothesis)
        return 100.0 * meteor_score.meteor_score(references=reference, hypothesis=hypothesis)

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

    def best_meteor_score(self, reference, hypothesis):
        pass

    def avg_meteor_score(self):
        pass



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

        evaluator = MeteorScoreMetric()
        bleu = evaluator._evaluating(reference=references, hypothesis=hypothesis, etype=etype)
        return bleu
