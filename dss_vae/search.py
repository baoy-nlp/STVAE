import os
import time

import torch
import torch.nn.functional as F

from .utils.files import load_docs, write_docs
from .utils.inputs import load_whole_tensor


def search_semantic_candidate(query_tensor, candidate_tensor, topk):
    score = F.cosine_similarity(query_tensor.unsqueeze(1), candidate_tensor, -1)
    return score.topk(topk)[1]  # batch_size, topK


def filter_syntactic_provider(query_tensor, candidate_tensor, m):
    if m > 0:
        score = 1 - F.cosine_similarity(query_tensor.unsqueeze(1), candidate_tensor, -1)
        return score.topk(m)[1]
    else:
        score = F.cosine_similarity(query_tensor.unsqueeze(1), candidate_tensor, -1)
        return score.topk(-m)[1]


def searched_inputs(input_sem, input_syn, corpus_sem, corpus_syn, topk=100, m=20):
    """

    :param input_sem: batch_size, latent_size
    :param input_syn:  batch_size,latent_size
    :param corpus_sem: corpus_size, latent_size
    :param corpus_syn: corpus_size, latent_size
    :param topk:
    :param m:
    :return:
        sen_ids: batch_size, k
        syn_searched_inputs: batch_size, k , latent_size
    """
    syn_candidate_ids = search_semantic_candidate(input_sem, corpus_sem, topk)  # batch_size,topK
    syn_candidate = F.embedding(syn_candidate_ids, corpus_syn)  # batch_size, topk, latent_size
    select_ids = filter_syntactic_provider(input_syn, syn_candidate, m)  # batch_size, k
    sen_ids = syn_candidate_ids.gather(1, select_ids)
    syn_inputs = syn_candidate.gather(1, select_ids.unsqueeze(-1).expand(-1, -1, syn_candidate.size(-1)))
    return sen_ids, syn_inputs


def searched_paraphrase(corpus_sen, sem_tensor_dict, syn_tensor_dict, batch_size, topk, m):
    eval_time = 0
    predicts = []
    corpus_sem = sem_tensor_dict['corpus']
    corpus_syn = syn_tensor_dict['corpus']

    test_sem = sem_tensor_dict['test']
    test_syn = syn_tensor_dict['test']

    sample_num = (test_sem.size(0) // batch_size) + int(test_sem.size(0) % batch_size != 0)

    for i in range(sample_num):
        input_sem = test_sem[batch_size * i:batch_size * (i + 1)]
        input_syn = test_syn[batch_size * i:batch_size * (i + 1)]
        with torch.no_grad():
            start_time = time.time()
            predict_ids, _ = searched_inputs(input_sem, input_syn, corpus_sem, corpus_syn, topk, m)
            eval_time += (time.time() - start_time)

        for single_ids in predict_ids:
            predict = [corpus_sen[ids.item()] for ids in single_ids]
            predicts.append(predict)

    return predicts, eval_time


def searched_beam_generate(model, sem_tensor_dict, syn_tensor_dict, batch_size, topk, m):
    eval_time = 0
    predicts = []
    corpus_sem = sem_tensor_dict['corpus']
    corpus_syn = syn_tensor_dict['corpus']

    test_sem = sem_tensor_dict['test']
    test_syn = syn_tensor_dict['test']

    src_num = test_sem.size(0)
    sample_num = (src_num // batch_size) + int(src_num % batch_size != 0)

    for s_id in range(sample_num):
        sem_z = test_sem[batch_size * s_id:batch_size * (s_id + 1)]
        syn_z = test_syn[batch_size * s_id:batch_size * (s_id + 1)]
        with torch.no_grad():
            start_time = time.time()
            _, syn_searched_z = searched_inputs(sem_z, syn_z, corpus_sem, corpus_syn, topk, m)
            bs, sample, latent_size = syn_searched_z.size()
            syn_extend_z = syn_searched_z.contiguous().view(bs * sample, -1)
            sem_extend_z = sem_z.unsqueeze(1).expand(-1, sample, -1).contiguous().view(bs * sample, -1)
            input_ret = {
                'batch_size': bs * sample,
                'sem_z': sem_extend_z,
                'syn_z': syn_extend_z,
            }
            predict_ids = model.latent_to_predict(input_ret, to_word=False)
            eval_time += (time.time() - start_time)
            predict = model.to_word(predict_ids)
            for b_id in range(bs):
                predicts.append(predict[b_id * sample:(b_id + 1) * sample])

    return predicts, eval_time


def searched_generate(model, sem_tensor_dict, syn_tensor_dict, batch_size, topk, m):
    eval_time = 0
    predicts = []
    corpus_sem = sem_tensor_dict['corpus']
    corpus_syn = syn_tensor_dict['corpus']

    test_sem = sem_tensor_dict['test']
    test_syn = syn_tensor_dict['test']

    src_num = test_sem.size(0)
    sample_num = (src_num // batch_size) + int(src_num % batch_size != 0)

    for s_id in range(sample_num):
        sem_z = test_sem[batch_size * s_id:batch_size * (s_id + 1)]
        syn_z = test_syn[batch_size * s_id:batch_size * (s_id + 1)]
        with torch.no_grad():
            start_time = time.time()
            _, syn_searched_z = searched_inputs(sem_z, syn_z, corpus_sem, corpus_syn, topk, m)
            bs = syn_searched_z.size(0)
            input_ret = {
                'batch_size': bs,
                'sem_z': sem_z,
                'syn_z': syn_searched_z.mean(1),
            }
            predict_ids = model.latent_to_predict(input_ret, to_word=False)
            eval_time += (time.time() - start_time)
            predict = model.to_word(predict_ids)
            predicts.extend(predict)

    return predicts, eval_time


def add_search_args(parser):
    parser.add_argument('--eval-topk', type=int, default=100)
    parser.add_argument('--eval-m', type=int, default=20)
    parser.add_argument('--search-func', type=str, default='paraphrase')
    return parser


def search_vae(args, file_dict, model, **kwargs):
    rest_dir = file_dict['out_dir']
    ref_ori = load_docs(fname=os.path.join(rest_dir, "{}.src.txt".format(args.eval_dataset)))
    ref_tgt = load_docs(fname=os.path.join(rest_dir, "{}.tgt.txt".format(args.eval_dataset)))

    output_dir = '{}/{}-K_{}-{}'.format(
        rest_dir,
        "Search" if args.search_func == "paraphrase" else (
                "Generate" + ("" if args.search_func == 'beam' else "-mean")),
        args.eval_topk,
        "U_{}".format(abs(args.eval_m)) if args.eval_m < 0 else "L_{}".format(abs(args.eval_m)),
    )

    sem_tensor_dict = {
        'corpus': load_whole_tensor(file=os.path.join(rest_dir, 'train.sem.txt')),
        'test': load_whole_tensor(file=os.path.join(rest_dir, "{}.sem.txt".format(args.eval_dataset)))

    }
    syn_tensor_dict = {
        'corpus': load_whole_tensor(file=os.path.join(rest_dir, 'train.syn.txt')),
        'test': load_whole_tensor(file=os.path.join(rest_dir, "{}.syn.txt".format(args.eval_dataset)))
    }

    if args.search_func == 'paraphrase':
        corpus_sent = load_docs(fname=os.path.join(rest_dir, "train.src.txt"))

        predicts, cum_time = searched_paraphrase(corpus_sent, sem_tensor_dict, syn_tensor_dict,
                                                 batch_size=args.eval_batch_size, topk=args.eval_topk, m=args.eval_m)
    elif args.search_func == 'beam':
        predicts, cum_time = searched_beam_generate(
            model, sem_tensor_dict, syn_tensor_dict, batch_size=args.eval_batch_size, topk=args.eval_topk, m=args.eval_m
        )
    else:
        predicts, cum_time = searched_generate(
            model, sem_tensor_dict, syn_tensor_dict, batch_size=args.eval_batch_size, topk=args.eval_topk, m=args.eval_m
        )

    from .metrics.paraphrase_metric import ParaphraseMetric
    ori_bleu, tgt_bleu = ParaphraseMetric.evaluate_transfer_generation(
        ref_ori, ref_tgt, predicts, output_dir
    )
    eval_ret = {'BLEU-ori': ori_bleu, 'BLEU-tgt': tgt_bleu, 'EVAL TIME': cum_time}

    for key, val in eval_ret.items():
        print("==\t%s: %0.3f\t==" % (key, val))

    write_docs(fname=os.path.join(output_dir, "score.txt"), docs=eval_ret)
