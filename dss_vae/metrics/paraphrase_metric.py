import os

from .bleu import BleuScorer
from .meteor import MeteorScorer


def write_docs(fname, docs):
    with open(fname, 'w') as f:
        for doc in docs:
            if isinstance(doc, list):
                doc = "\t".join(doc)
            f.write(str(doc).strip())
            f.write('\n')


def format_output(val, keep_num=3):
    if isinstance(val, str):
        return val
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return "%0.2f" % val
    if isinstance(val, list) or isinstance(val, tuple):
        return " ".join([format_output(item) for item in val])


class ParaphraseMetric(object):
    def __init__(self):
        self.name = 'Paraphrase-Metrics'

    @classmethod
    def preprocess(cls, **kwargs):
        BleuScorer.preprocess(**kwargs)

    @classmethod
    def evaluate_bleu(cls, ori_file, ref_file, pred_files, etype='corpus', index_select=4):
        bleu_ori = BleuScorer.evaluate_file(
            pred_file=pred_files, gold_files=ori_file, score_type=etype,
            index_select=index_select
        )
        bleu_tgt = BleuScorer.evaluate_file(
            pred_file=pred_files, gold_files=ref_file, score_type=etype,
            index_select=index_select
        )
        return bleu_ori, bleu_tgt

    @classmethod
    def _select_best_candidate(cls, predicts, ref_tgt, score_func="bleu"):
        selected = []
        scorer = BleuScorer if score_func == 'bleu' else MeteorScorer
        for predict_strs, gold_strs in zip(predicts, ref_tgt):
            scores = scorer.compute_beam_score(predict_strs, gold_strs)
            selected.append(predict_strs[scores.index(max(scores))])
        return selected

    @classmethod
    def evaluate_quality(cls, ref_ori, ref_tgt, predicts, rest_dir='tmp', etype='corpus'):
        """
        include:
            - best-ori-bleu
            - best-tgt-ble
            - avg-ori-bleu
            - avg-tgt-bleu
            - meteor score
        """
        if not os.path.exists(rest_dir):
            os.makedirs(rest_dir)

        write_docs(fname=os.path.join(rest_dir, 'ori.txt'), docs=ref_ori)
        write_docs(fname=os.path.join(rest_dir, 'para.txt'), docs=ref_tgt)

        if isinstance(predicts[0], list):
            write_docs(fname=os.path.join(rest_dir, 'pred.beam.txt'), docs=predicts)

            avg_ori = BleuScorer.evaluate_average_score(predicts, ref_ori, index_select=0)
            avg_tgt = BleuScorer.evaluate_average_score(predicts, ref_tgt, index_select=0)
        else:
            write_docs(fname=os.path.join(rest_dir, 'pred.txt'), docs=predicts)
            avg_ori, avg_tgt = cls.evaluate_bleu(
                ori_file=os.path.join(rest_dir, 'ori.txt'),
                ref_file=os.path.join(rest_dir, 'para.txt'),
                pred_files=os.path.join(rest_dir, 'pred.txt'),
                etype=etype,
                index_select=0
            )

        avg_meteor_score = MeteorScorer.evaluate_average_score(predicts, ref_tgt)
        max_bleu_ori, max_bleu_tgt = avg_ori, avg_tgt

        if isinstance(predicts[0], list):
            bleu_select = cls._select_best_candidate(predicts, ref_tgt, score_func='bleu')
            write_docs(fname=os.path.join(rest_dir, 'pred.best-bleu.txt'), docs=bleu_select)
            max_bleu_ori, max_bleu_tgt = cls.evaluate_bleu(
                ori_file=os.path.join(rest_dir, 'ori.txt'),
                ref_file=os.path.join(rest_dir, 'para.txt'),
                pred_files=os.path.join(rest_dir, 'pred.best-bleu.txt'),
                etype=etype,
                index_select=0
            )

            # meteor_select = cls._select_best_candidate(predicts, ref_tgt, score_func="meteor")
            # write_docs(fname=os.path.join(rest_dir, 'pred.best-meteor.txt'), docs=meteor_select)
            # max_meteor_ori, max_meteor_tgt = cls.evaluate_bleu(
            #     ori_file=os.path.join(rest_dir, 'ori.txt'),
            #     ref_file=os.path.join(rest_dir, 'para.txt'),
            #     pred_files=os.path.join(rest_dir, 'pred.best-meteor.txt'),
            #     etype=etype,
            #     index_select=0
            # )

        return {
            "Avg-ORI": avg_ori,
            "Avg-TGT": avg_tgt,
            "Avg-Meteor": avg_meteor_score,
            "BLEU-S-ORI": max_bleu_ori,
            "BLEU-S-TGT": max_bleu_tgt,
            # "METEORS-ORI": max_bleu_ori,
            # "METEORS-TGT": max_bleu_tgt,
        }

    @classmethod
    def evaluate_diversity(cls, predicts, rest_dir='tmp'):
        if not os.path.exists(rest_dir):
            os.makedirs(rest_dir)

        if isinstance(predicts[0], list):
            write_docs(fname=os.path.join(rest_dir, 'pred_beam.txt'), docs=predicts)
            pairwise_ret = BleuScorer.evaluate_pairwise_score(predicts, index=0)
        else:
            pairwise_ret = [0, 0, 0, 0]

        return {
            'Self-BLEU': format_output(pairwise_ret, 2),
        }

    # the below item will be removed

    @classmethod
    def evaluate_transfer_generation(cls, ref_ori, ref_tgt, predicts, rest_dir='tmp', etype='corpus'):
        if not os.path.exists(rest_dir):
            os.makedirs(rest_dir)

        write_docs(fname=os.path.join(rest_dir, 'ref_ori.txt'), docs=ref_ori)
        write_docs(fname=os.path.join(rest_dir, 'ref_tgt.txt'), docs=ref_tgt)

        if isinstance(predicts[0], list):
            write_docs(fname=os.path.join(rest_dir, 'pred_beam.txt'), docs=predicts)
            selects = cls._select_best_candidate(predicts, ref_tgt)
            write_docs(fname=os.path.join(rest_dir, 'pred.txt'), docs=selects)
        else:
            write_docs(fname=os.path.join(rest_dir, 'pred.txt'), docs=predicts)

        return cls.evaluate_bleu(
            ori_file=os.path.join(rest_dir, 'ref_ori.txt'),
            ref_file=os.path.join(rest_dir, 'ref_tgt.txt'),
            pred_files=os.path.join(rest_dir, 'pred.txt'),
            etype=etype
        )
