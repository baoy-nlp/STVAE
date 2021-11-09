"""
Preprocess the file, and compute the bleu scorer, including:

- Corpus BLEU and Sentence BLEU, with four category (basically):
    - BLEU-1
    - BLEU-2
    - BLEU-3
    - BLEU-4
- Pairwise BLEU scores (list of inputs)
    - default is one-to-multi-reference scores
"""

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

from .base_metric import BaseMetric

bleu_score_weights = {
    1: (1.0, 0.0, 0.0, 0.0),
    2: (0.5, 0.5, 0.0, 0.0),
    3: (0.34, 0.33, 0.33, 0.0),
    4: (0.25, 0.25, 0.25, 0.25),
}


class BleuScorer(BaseMetric):
    def __init__(self):
        super(BleuScorer, self).__init__(name="BLEU")

    @classmethod
    def preprocess(cls, reference, hypothesis):
        if not (
                isinstance(reference, list)
                and isinstance(reference[0], list)
                and isinstance(reference[0][0], list)
                and isinstance(reference[0][0][0], str)
        ):
            raise RuntimeError(
                "reference type is not true, expect to be list(list(list(str))) but is {}".format(type(reference)))

        if not (
                isinstance(hypothesis, list)
                and isinstance(hypothesis[0], list)
                and isinstance(hypothesis[0][0], str)
        ):
            raise RuntimeError(
                "hypothesis type is not true, expect to be list(list(str)) but is {}".format(type(hypothesis)))

    def _evaluating(self, reference, hypothesis, index=0, etype="corpus"):
        self.preprocess(reference, hypothesis)  # preprocess to evaluate the BLEU scores
        return self.evaluate_score(
            reference=reference, hypothesis=hypothesis, index_select=index, etype=etype
        )

    @classmethod
    def evaluate_score(cls, reference, hypothesis, index_select=0, etype="corpus"):
        scorer = {
            'corpus': cls._corpus_bleu,
            'sents': cls._sentence_bleu
        }[etype]
        if index_select > 0:
            return scorer(references=reference, hypotheses=hypothesis, index_select=index_select)
        else:
            scores = [
                scorer(references=reference, hypotheses=hypothesis, index_select=i)
                for i in range(1, 5)
            ]
            return scores

    @classmethod
    def _single_sentence_bleu(cls, references, hypotheses, index_select=4):
        return sentence_bleu(references, hypotheses, weights=bleu_score_weights[index_select]) * 100.0

    @classmethod
    def _sentence_bleu(cls, references, hypotheses, index_select=4):
        list_score = [sentence_bleu(refs, hypo, weights=bleu_score_weights[index_select]) * 100.0 for refs, hypo in
                      zip(references, hypotheses)]
        return sum(list_score) / len(list_score)

    @classmethod
    def _corpus_bleu(cls, references, hypotheses, index_select=4):
        return corpus_bleu(
            list_of_references=references,
            hypotheses=hypotheses,
            weights=bleu_score_weights[index_select]
        ) * 100.0

    @classmethod
    def evaluate_file(cls, pred_file, gold_files, index_select=4, score_type="corpus"):
        # process file to candidates: single predict and multi targets
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

        bleu = cls.evaluate_score(
            reference=references,
            hypothesis=hypothesis,
            index_select=index_select,
            etype=score_type
        )
        return bleu

    @classmethod
    def evaluate_pairwise_score(cls, predicts, index=4):
        if index > 0:
            score = sum([cls.compute_pairwise_score(predict, index) for predict in predicts]) / len(predicts)
            return score
        else:
            scores = [
                sum([cls.compute_pairwise_score(predict, i) for predict in predicts]) / len(predicts)
                for i in range(1, 5)
            ]
            return scores

    @classmethod
    def compute_pairwise_score(cls, predicts, index=4):
        """
        compute inter bleu-index
        """
        predict_list = [predict.split(" ") for predict in predicts]
        bleu_list = [
            cls._single_sentence_bleu(
                references=predict_list[:i] + predict_list[i + 1:],
                hypotheses=predict_list[i],
                index_select=index
            ) for i in range(len(predicts))
        ]
        return sum(bleu_list) / len(predicts)

    @classmethod
    def evaluate_average_score(cls, predicts, ref_tgt, index_select=4):

        def compute_score(i_select):
            _scores = []
            for predict_strs, gold_strs in zip(predicts, ref_tgt):
                beam_scores = cls.compute_beam_score(predict_strs, gold_strs, index_select=i_select)
                avg_scores = sum(beam_scores) / len(beam_scores)
                _scores.append(avg_scores)
            return sum(_scores) / len(_scores)

        if index_select > 0:
            return compute_score(index_select)
        else:
            scores = [
                compute_score(i) for i in range(1, 5)
            ]
            return scores

    @classmethod
    def compute_beam_score(cls, predict_strs, ref_strs, index_select=4):
        """
        :param predict_strs: list(str) or str
        :param ref_strs: list(str) or str
        :param index_select: BLEU-I
        :return:
        """

        if not isinstance(ref_strs, list):
            ref_strs = [ref_strs, ]
        if not isinstance(predict_strs, list):
            predict_strs = [predict_strs, ]

        reference = [ref.split(" ") for ref in ref_strs]
        predicts = [predict.split(" ") for predict in predict_strs]

        return [cls._single_sentence_bleu(reference, predict, index_select) for predict in predicts]


hypo1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
hypo2 = "It is to insure the troops forever hearing the activity guidebook that party direct"
ref1 = "It is a guide to action that ensures that the military will forever heed Party commands"
ref2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party"
ref3 = "It is the practical guide for the army always to heed the directions of the party"
