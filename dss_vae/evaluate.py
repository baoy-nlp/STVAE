import os

from dss_vae.generator.syntax_generator import SyntaxGenerator
from dss_vae.generator.vae_generator import VAEGenerator
from dss_vae.metrics import BleuScorer
from dss_vae.metrics import ParaphraseMetric
from dss_vae.utils.files import write_docs


def corpus_elbo(iterator, model, criterion, step=-1, **kwargs):
    """ evaluate the ELBOs """
    generator = VAEGenerator(model)
    return generator.elbo(iterator, criterion, step)


def eval_vae(iterator, model, criterion, rest_dir='tmp', step=-1, **kwargs):
    """ evaluate the VAEs """
    bleu_dict = reconstruct(iterator, model, rest_dir)
    elbo_dict = corpus_elbo(iterator, model, criterion, step)
    elbo_dict['BLEU'] = bleu_dict['BLEU']
    elbo_dict['ELBO'] = elbo_dict['ELBO'].mean().item()
    return elbo_dict


def reconstruct(iterator, model, rest_dir="tmp", **kwargs):
    generator = VAEGenerator(model)
    reference, predicts, cum_time = generator.reconstruct(iterator)
    if not os.path.exists(rest_dir):
        os.makedirs(rest_dir)

    write_docs(fname=os.path.join(rest_dir, 'ref.txt'), docs=reference)
    write_docs(fname=os.path.join(rest_dir, 'pref.txt'), docs=predicts)

    bleu_score = BleuScorer.evaluate_file(
        pred_file=os.path.join(rest_dir, 'pref.txt'),
        gold_files=os.path.join(rest_dir, 'ref.txt')
    )
    return {
        'BLEU': bleu_score,
        'EVAL TIME': cum_time
    }


def sampling(model, rest_dir='tmp', sample_size=10000, **kwargs):
    generator = VAEGenerator(model)
    predicts, cum_time = generator.generate(sample_size)
    if not os.path.exists(rest_dir):
        os.makedirs(rest_dir)

    write_docs(fname=os.path.join(rest_dir, 'sample.txt'), docs=predicts)

    return {'EVAL TIME': cum_time}


def transfer(iterator, model, rest_dir='tmp', **kwargs):
    generator = SyntaxGenerator(model)
    ref_ori, ref_tgt, predicts, cum_time = generator.syntactic_transferring(testset=iterator)
    ori_bleu, tgt_bleu = ParaphraseMetric.evaluate_transfer_generation(ref_ori, ref_tgt, predicts, rest_dir)
    return {'BLEU-sem': ori_bleu, 'BLEU-syn': tgt_bleu, 'EVAL TIME': cum_time}


def paraphrase(iterator, model, rest_dir='tmp', **kwargs):
    generator = SyntaxGenerator(model)
    reference_ori, reference_tgt, predicts, test_time = generator.paraphrasing(testset=iterator, **kwargs)
    quality = ParaphraseMetric.evaluate_quality(reference_ori, reference_tgt, predicts, rest_dir=rest_dir)
    diversity = ParaphraseMetric.evaluate_diversity(predicts, rest_dir=rest_dir)
    ret = quality
    ret.update(diversity)
    ret["EVAL TIME"] = test_time
    return quality


def meta_paraphrase(iterator, model, rest_dir='tmp', **kwargs):
    generator = SyntaxGenerator(model)
    ref_ori, ref_tgt, predicts, cum_time = generator.gmm_paraphrasing(testset=iterator, **kwargs)
    ori_bleu, tgt_bleu = ParaphraseMetric.evaluate_transfer_generation(ref_ori, ref_tgt, predicts, rest_dir)
    return {'BLEU-ori': ori_bleu, 'BLEU-tgt': tgt_bleu, 'EVAL TIME': cum_time}
