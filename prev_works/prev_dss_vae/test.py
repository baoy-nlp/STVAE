from __future__ import absolute_import

import os
import sys

import torch

sys.path.append("..")
from prev_works.prev_dss_vae import batchify_examples
from prev_works.prev_dss_vae import evaluate_all
from prev_works.prev_dss_vae.metrics.vae_metrics import SyntaxVaeEvaluator
from prev_works.prev_dss_vae import BaseSeq2seq
from prev_works.prev_dss_vae.utils.model_utils import get_eval_info
from prev_works.prev_dss_vae.utils.model_utils import load_model
from prev_works.prev_dss_vae.utils.utility import load_docs
from prev_works.prev_dss_vae import Dataset
from prev_works.prev_dss_vae import to_example

FILE_DICT = {
    # "PRED": "trans.length.txt.unmatch.pred.par.s2b",
    # "ORI": "trans.length.txt.unmatch.ori.par.s2b",
    # "TGT": "trans.length.txt.unmatch.tgt.par.s2b"
    "PRED": "trans.length.txt.unmatch.pred.par",
    "ORI": "trans.length.txt.unmatch.ori.par",
    "TGT": "trans.length.txt.unmatch.tgt.par"
}


def testing(ptrn_dir, test_dir, model=None):
    # main_args, model_args = load_configs(ptrn_dir)
    # test_model = load_model(main_args, model_args, check_dir=False)
    # ori_pred_dis = evaluate_syntax_similarity(
    #     model=test_model,
    #     oracle_tree_file=os.path.join(test_dir, FILE_DICT['ORI']),
    #     pred_tree_file=os.path.join(test_dir, FILE_DICT['PRED'])
    # )
    # tgt_pred_dis = evaluate_syntax_similarity(
    #     model=test_model,
    #     oracle_tree_file=os.path.join(test_dir, FILE_DICT['TGT']),
    #     pred_tree_file=os.path.join(test_dir, FILE_DICT['PRED'])
    # )
    ori_pred_dis = evaluate_all(
        ori=os.path.join(test_dir, FILE_DICT['ORI']),
        pred=os.path.join(test_dir, FILE_DICT['PRED']),
        tgt=os.path.join(test_dir, FILE_DICT['TGT'])
    )
    # print("Sim To Ori", sum(ori_pred_dis)/len(ori_pred_dis))
    # print("Sim To Tgt", sum(tgt_pred_dis)/len(tgt_pred_dis))


def dot_product_distance(inputs_a, inputs_b):
    """

    Args:
        inputs_a: [batch,hidden]
        inputs_b: [batch,hidden]

    Returns:

    """
    return (inputs_a * inputs_b).sum(dim=-1)


def evaluate_syntax_similarity(
        model: BaseSeq2seq,
        oracle_tree_file,
        pred_tree_file,
        batch_size=25,
        distance_type="cos"
):
    gold_trees = load_docs(oracle_tree_file)
    pred_trees = load_docs(pred_tree_file)
    gold_examples = [to_example(tree) for tree in gold_trees]
    pred_examples = [to_example(tree) for tree in pred_trees]

    # distance_func = torch.nn.CosineSimilarity(dim=-1, eps=1e-9) if distance_type == 'cos' else None
    distance_func = dot_product_distance
    batch_gold_examples, _ = batchify_examples(
        examples=gold_examples,
        sort=False,
        batch_size=batch_size
    )
    batch_pred_examples, _ = batchify_examples(
        examples=pred_examples,
        sort=False,
        batch_size=batch_size
    )
    sum_instance = 0.0
    sum_distance = 0.0
    # for batch_gold, batch_pred in zip(batch_gold_examples, batch_pred_examples):
    #     gold_hidden = model.get_hidden(examples=batch_gold)
    #     pred_hidden = model.get_hidden(examples=batch_pred)
    #     sum_instance += len(batch_gold)
    #     sum_distance += distance_func(gold_hidden, pred_hidden).sum()
    for batch_gold, batch_pred in zip(gold_examples, pred_examples):
        gold_hidden = model.get_hidden(examples=batch_gold)
        pred_hidden = model.get_hidden(examples=batch_pred)
        sum_instance += len(batch_gold)
        sum_distance += distance_func(gold_hidden, pred_hidden).sum()
    avg_distance = sum_distance.item() / sum_instance
    print("average distance:", avg_distance)
    return avg_distance


PLOT_NUM = 5000


def to_docs(example_batch, vec_ret, file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_dir + "/words_{}.txt".format(PLOT_NUM), "a") as f:
        for example in example_batch:
            f.write(" ".join(example.src))
            f.write("\n")
    with open(file_dir + "/sem_{}.txt".format(PLOT_NUM), "a") as f:
        sem_vec = vec_ret['sem_z']
        for row_index in range(sem_vec.size(0)):
            f.write("\t".join(str(sem_vec[row_index].data.tolist())[1:-1].split(",")))
            f.write("\n")
    with open(file_dir + "/syn_{}.txt".format(PLOT_NUM), "a") as f:
        syn_vec = vec_ret['syn_z']
        for row_index in range(syn_vec.size(0)):
            f.write("\t".join(str(syn_vec[row_index].data.tolist())[1:-1].split(",")))
            f.write("\n")


def analysis_space(model, dataset, args):
    model.eval()
    word_result = []
    vec_result = []
    dataset.examples = dataset.examples[:PLOT_NUM]
    for examples in dataset.batch_iter(batch_size=args.batch_size, shuffle=False):
        # word_result.append(examples)
        ret = model.extract_variable(examples)
        to_docs(examples, ret, file_dir=args.output_path)
        torch.cuda.empty_cache()
        # vec_result.append(ret)
    print("== extract finish ==")


def test_vae(main_args, model_args, args):
    model = load_model(main_args, model_args, check_dir=False)
    out_dir = get_eval_info(main_args=main_args, model_args=model_args, mode="Test")
    model.eval()
    if not os.path.exists(out_dir):
        sys.exit(-1)
    if model_args.model_select.startswith("Origin"):
        model_args.eval_bs = 20 if model_args.eval_bs < 20 else model_args.eval_bs
    evaluator = SyntaxVaeEvaluator(
        model=model,
        out_dir=out_dir,
        batch_size=model_args.eval_bs,
        train_batch_size=main_args.batch_size
    )

    para_eval_dir = getattr(
        main_args, "eval_dir", "/home/user_data/baoy/projects/seq2seq_parser/data/quora-mh/unsupervised"
    )
    para_eval_list = getattr(main_args, "eval_list", ["dev.para.txt"])
    input_mode = args.eval_mode
    setattr(args, "batch_size", main_args.batch_size)
    if input_mode == 0:
        train_set = Dataset.from_bin_file(main_args.train_file)
        print("======== Reconstructing Valid Set ========")
        test_set = Dataset.from_bin_file(main_args.dev_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="dev")
        print("finish")
        print("======== Reconstructing Test Set =======")
        test_set = Dataset.from_bin_file(main_args.test_file)
        evaluator.evaluate_reconstruction(examples=test_set.examples, eval_desc="test")
        print("finish")
        print("======== Sampling Generation =======")
        evaluator.evaluate_generation(corpus_examples=train_set.examples, sample_size=len(test_set.examples),
                                      eval_desc="gen")
        print("finish")
    elif input_mode == 1:
        train_set = Dataset.from_bin_file(main_args.train_file)
        print("======== Sampling Generation =======")
        test_exam = Dataset.from_bin_file(main_args.test_file).examples
        evaluator.evaluate_generation(corpus_examples=train_set.examples, sample_size=len(test_exam), eval_desc="gen")
        print("finish")
    elif input_mode == 2:
        print("======== Paraphrase Generation ========")
        evaluator.evaluate_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 3:
        print("======== Style-Controlled Generation ========")
        evaluator.evaluate_control(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 4:
        trans_eval_list = ["trans.length.txt", "trans.random.txt"]
        print("======== Style-Transfer Generation ========")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=trans_eval_list, eval_desc="unmatch")
        evaluator.evaluate_style_transfer(eval_dir=para_eval_dir, eval_list=para_eval_list, eval_desc="match")
        print("finish")
    elif input_mode == 5:
        print("======== Random Syntax Select ========")
        evaluator.evaluate_pure_para(eval_dir=para_eval_dir, eval_list=para_eval_list)
        print("finish")
    elif input_mode == 6:
        print("======== Extract Embedding for Visualization ========")
        test_set = Dataset.from_bin_file(main_args.train_file, remove_same=True)
        analysis_space(model, test_set, args)
    elif input_mode == 7:
        print("======== Case Study for DSS-VAE ========")
        ref_sem = input("ref_sem: ")
        while not ref_sem.startswith("EXIT"):
            e = to_example(ref_sem)
            words = model.predict(e)
            print("origin:", " ".join(words[0][0][0]))
            ref_syn = input("ref syn : ")
            while not ref_syn.startswith("NEXT"):
                syn_ref = to_example(ref_syn)
                ret = model.eval_adv(e, syn_ref)
                # if not model_args.model_select == "OriginVAE":
                # print("ref syntax: ", " ".join(ret['ref syn'][0][0][0]))
                # print("ori syntax: ", " ".join(ret['ori syn'][0][0][0]))
                print("switch result: ", " ".join(ret['res'][0][0][0]))
                ref_syn = input("ref syn: ")
            print("============= NEXT =============")
            ref_sem = input("ref_sem : ")
    elif input_mode == 8:
        print("======== Case Study for UPG ========")
        ref_sem = input("ref_sem: ")
        while not ref_sem.startswith("EXIT"):
            e = to_example(ref_sem)
            words = model.predict(e)
            print("origin:", " ".join(words[0][0][0]))
            ret = model.conditional_generating(condition='sem', examples=e)
            print("UPG: ", " ".join(ret['res'][0][0][0]))
            print("============= NEXT =============")
            ref_sem = input("ref_sem : ")
    elif input_mode == 9:
        print("======== Case Study for Interpolating ========")
        ref_sem = input("ref_sem: ")
        while not ref_sem.startswith("EXIT"):
            e = to_example(ref_sem)
            words = model.predict(e)
            print("origin:", " ".join(words[0][0][0]))
            ref_syn = input("ref syn : ")
            while not ref_syn.startswith("NEXT"):
                syn_ref = to_example(ref_syn)
                ret = model.interploate_for_syntax(e, syn_ref)
                # if not model_args.model_select == "OriginVAE":
                # print("ref syntax: ", " ".join(ret['ref syn'][0][0][0]))
                # print("ori syntax: ", " ".join(ret['ori syn'][0][0][0]))
                for i, item in enumerate(ret):
                    print("switch result {}: ".format(i), " ".join(item[0][0][0]))
                ref_syn = input("ref syn: ")
            print("============= NEXT =============")
            ref_sem = input("ref_sem : ")
    else:
        raw = input("raw sent: ")
        while not raw.startswith("EXIT"):
            e = to_example(raw)
            words = model.predict(e)
            print("origin:", " ".join(words[0][0][0]))
            ref_syn = input("ref syn : ")
            while not ref_syn.startswith("NEXT"):
                syn_ref = to_example(ref_syn)
                ret = model.eval_adv(e, syn_ref)
                if not model_args.model_select == "OriginVAE":
                    print("ref syntax: ", " ".join(ret['ref syn'][0][0][0]))
                    print("ori syntax: ", " ".join(ret['ori syn'][0][0][0]))
                print("switch result: ", " ".join(ret['res'][0][0][0]))
                ref_syn = input("ref syn: ")
            raw = input("input : ")
