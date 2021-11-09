import fractions
import math
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import closest_ref_length, brevity_penalty, modified_precision, SmoothingFunction

try:
    fractions.Fraction(denominator=1000, _normalize=False)
    from fractions import Fraction
except TypeError:
    from nltk.compat import Fraction


def sentence_bleu(
        references,
        hypothesis,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
        emulate_multibleu=False,
        details=False,
        index_select=0,
):
    return corpus_bleu(
        [references],
        [hypothesis],
        weights, smoothing_function, auto_reweigh,
        emulate_multibleu,
        details=details,
        index_select=index_select
    )


def corpus_bleu(
        list_of_references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=None,
        auto_reweigh=False,
        emulate_multibleu=False,
        details=False,
        index_select=0,
):
    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0

    references, hypothesis, hyp_len = None, None, None

    res = 0, np.array([0, 0, 0, 0, 0]), 0, 0, 0

    if len(list_of_references) != len(hypotheses):
        print("The number of hypotheses and their reference(s) should be the same")
        if details:
            return res
        else:
            return res[1][index_select]

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator

        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)

    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Uniformly re-weighting based on maximum hypothesis lengths if largest
    # order of n-grams < 4 and weights is set at default.
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths

    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]

    # Collects the various precision values for the different ngram orders.
    s_detail = [xx.numerator / xx.denominator * 100 for xx in p_n]

    if p_numerators[1] == 0:
        if details:
            return res
        else:
            return res[1][index_select]

    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0

    p_n = smoothing_function(
        p_n,
        references=references,
        hypothesis=hypothesis,
        hyp_len=hyp_len,
        emulate_multibleu=emulate_multibleu
    )
    bleu = (w * math.log(p_i) for i, (w, p_i) in enumerate(zip(weights, p_n)))
    bleu = bp * math.exp(math.fsum(bleu)) * 100
    bleu = round(bleu, 4) if emulate_multibleu else bleu

    res = bleu, np.array([bleu] + s_detail), bp, ref_lengths, hyp_lengths
    if details:
        return res
    else:
        return res[1][index_select]
