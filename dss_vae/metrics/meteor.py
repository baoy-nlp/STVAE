"""
Preprocess the file, and compute the meteor scorer, including:

- Meteor score (basically)
- Beam Meteor Scorer
"""

from nltk.translate import meteor_score

from .base_metric import BaseMetric


class MeteorScorer(BaseMetric):
    def __init__(self):
        super(MeteorScorer, self).__init__(name="METEOR")

    @classmethod
    def preprocess(cls, reference, hypothesis):
        """

        :param reference: list(list(str))
        :param hypothesis: list(str)
        :return:
        """
        if not (
                isinstance(reference, list)
                and isinstance(reference[0], list)
                and isinstance(reference[0][0], str)
        ):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(str)) but is {}".format(type(reference)))

        if not (
                isinstance(hypothesis, list)
                and isinstance(hypothesis[0], str)
        ):
            raise RuntimeError(
                "hypothesis type is not true, expect to be list(str) but is {}".format(type(hypothesis)))

    def _evaluating(self, reference, hypothesis, **kwargs):
        self.preprocess(reference=reference, hypothesis=hypothesis)
        return self.evaluate_score(reference, hypothesis)

    @classmethod
    def _single_meteor_scores(cls, reference, hypothesis):
        return 100.0 * meteor_score.meteor_score(references=reference, hypothesis=hypothesis)

    @classmethod
    def evaluate_score(cls, reference, hypothesis):
        list_score = [
            cls._single_meteor_scores(refs, hypo) for refs, hypo in zip(reference, hypothesis)
        ]
        return sum(list_score) / len(hypothesis)

    @classmethod
    def evaluate_file(cls, pred_file, gold_files, **kwargs):
        with open(pred_file, "r") as f:
            hypothesis = [line.strip() for line in f.readlines()]

        references = [[] for _ in range(len(hypothesis))]
        if not isinstance(gold_files, list):
            gold_files = [gold_files, ]

        for file in gold_files:
            with open(file, "r") as f:
                ref_i = [line.strip() for line in f.readlines()]
                for idx, ref in enumerate(ref_i):
                    references[idx].append(ref)

        meteor = cls.evaluate_score(reference=references, hypothesis=hypothesis)
        return meteor

    @classmethod
    def evaluate_average_score(cls, predicts, ref_tgt):
        scores = []
        for predict_strs, gold_strs in zip(predicts, ref_tgt):
            beam_scores = cls.compute_beam_score(predict_strs, gold_strs)
            avg_scores = sum(beam_scores) / len(beam_scores)
            scores.append(avg_scores)
        return sum(scores) / len(scores)

    @classmethod
    def compute_beam_score(cls, predicts, reference):
        """

        :param predicts: list(str) or str
        :param reference: list(str)
        :return:
        """
        if not isinstance(reference, list):
            reference = [reference, ]
        if not isinstance(predicts, list):
            predicts = [predicts, ]

        return [cls._single_meteor_scores(reference, predict) for predict in predicts]
