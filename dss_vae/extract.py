import os
import time

import numpy as np


def extract_syntax(iterator, model, rest_dir='tmp', dataset_name='valid', **kwargs):
    """ evaluate the syntax-transfer generation """
    cum_time, num_valid = 0, 0
    print('Extract %s' % dataset_name)
    if not os.path.exists(rest_dir):
        os.makedirs(rest_dir)

    src_file = os.path.join(rest_dir, '{}.src.txt'.format(dataset_name))
    tgt_file = os.path.join(rest_dir, '{}.tgt.txt'.format(dataset_name))
    syn_file = os.path.join(rest_dir, '{}.syn.txt'.format(dataset_name))
    sem_file = os.path.join(rest_dir, '{}.sem.txt'.format(dataset_name))

    with open(src_file, "w") as src_f, open(tgt_file, "w") as tgt_f, open(sem_file, "w") as sem_f, open(syn_file,
                                                                                                        "w") as syn_f:
        for batch in iterator:
            num_valid += len(batch)
            start_time = time.time()
            ret = model.extract_latent(batch, use_tgt=False)
            sem_z = ret['sem'].cpu().detach().numpy()
            syn_z = ret['syn'].cpu().detach().numpy()
            batch_time = time.time() - start_time
            cum_time += batch_time
            srcs = [" ".join(e.src) for e in batch]
            tgts = [" ".join(e.tgt) for e in batch]

            for index, src in enumerate(srcs):
                src_f.write(src)
                src_f.write("\n")

                tgt_f.write(tgts[index])
                tgt_f.write("\n")

                sem = np.array2string(sem_z[index], precision=6, max_line_width=2000)
                sem_f.write(sem)
                sem_f.write("\n")

                syn = np.array2string(syn_z[index], precision=6, max_line_width=2000)
                syn_f.write(syn)
                syn_f.write("\n")

    return {
        'Dataset Size': num_valid,
        'EVAL TIME': cum_time
    }


def extract_template(iterator, model, rest_dir='tmp', dataset_name='valid', **kwargs):
    """ evaluate the syntax template """
    cum_time, num_valid = 0, 0
    print('Extract %s' % dataset_name)
    if not os.path.exists(rest_dir):
        os.makedirs(rest_dir)

    # src_file = os.path.join(rest_dir, '{}.src.txt'.format(dataset_name))
    st_file = os.path.join(rest_dir, '{}.sent_t.txt'.format(dataset_name))
    ts_file = os.path.join(rest_dir, '{}.t_sent.txt'.format(dataset_name))
    tscore_file = os.path.join(rest_dir, '{}.t_score.txt'.format(dataset_name))

    iter = 0
    sentences = []
    template_scores = []
    template_nums = []

    for batch in iterator:
        iter += 1
        print("\rFinish Iter:{}".format(iter), end="")
        num_valid += len(batch)
        start_time = time.time()
        ret = model.extract_latent(batch, use_tgt=False)
        t_score = ret['syn_t_score'].cpu().detach().numpy()
        t_num = ret['syn_t_num'].cpu().detach().numpy()
        batch_time = time.time() - start_time
        cum_time += batch_time
        srcs = [" ".join(e.src) for e in batch]

        for index, src in enumerate(srcs):
            sentences.append(src)
            template_scores.append(t_score[index].item())
            template_nums.append(t_num[index].item())

    template_sents = [[] for _ in range(max(template_nums)+1)]
    with open(st_file, "w") as f:
        for sent, num in zip(sentences, template_nums):
            line = "{}\t{}".format(str(num), sent)
            template_sents[num].append(sent)
            f.write(line)
            f.write("\n")

    with open(tscore_file, "w") as f:
        for score, num in zip(template_scores, template_nums):
            line = "{}\t{}".format(str(num), str(score))
            f.write(line)
            f.write("\n")

    with open(ts_file, "w") as f:
        for num, sents in enumerate(template_sents):
            for sent in sents:
                line = "{}\t{}".format(str(num), sent)
                f.write(line)
                f.write("\n")


    # with open(src_file, "w") as src_f, open(syn_file, "w") as score_f, open(temp_file, "w") as temp_f:

    # src_f.write(src)
    # src_f.write("\n")
    #
    # # score = np.array2string(t_score[index], precision=6, max_line_width=2000)
    # score = str(t_score[index].item())
    # score_f.write(score)
    # score_f.write("\n")
    #
    # # num = np.array2string(t_num[index], precision=1, max_line_width=2000)
    # num = str(t_num[index].item())
    # temp_f.write(num)
    # temp_f.write("\n")

    print("\n")
    print("Finish Extract!")
    print("Write Path:{}".format(rest_dir))

    return {
        'Dataset Size': num_valid,
        'EVAL TIME': cum_time
    }


eval_func_dict = {
    'latent': extract_syntax,
    'template': extract_template
}


def extract_vae(args, model, datasets, **kwargs):
    iterator = datasets[args.eval_dataset]
    extract_func = eval_func_dict[args.eval_func]

    eval_ret = extract_func(iterator, model, dataset_name=args.eval_dataset, **kwargs)

    for key, val in eval_ret.items():
        print("==\t%s: %0.3f\t==" % (key, val))
