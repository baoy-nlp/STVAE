def load_docs(fname):
    docs = []
    with open(fname, "r") as f:
        for line in f.readlines():
            docs.append(line.strip())
    return docs


def write_docs(fname, docs):
    docs = []
    with open(fname, "w") as f:
        for line in docs:
            f.write(line)
            f.write("\n")
    return docs


def combine_outputs(input_file, *ofnames):
    competitor_outputs = []
    for fname in ofnames:
        competitor_outputs.append(load_docs(fname))
    return load_docs(input_file), competitor_outputs


def record_the_shuffle(inputs, competitor_outputs):
    import random
    number_competitors = len(competitor_outputs)
    index_arr = list(range(number_competitors))

    shuffle_records = []

    shuffle_outputs = []

    for sent_ids in range(len(inputs)):
        random.shuffle(index_arr)
        shuffle_records.extend(index_arr)

        shuffle_outputs.append("=========Case:{}=========".format(sent_ids))
        shuffle_outputs.append("input:{}".format(inputs[sent_ids]))
        shuffle_outputs.append("-------------------------")
        for i in index_arr:
            shuffle_outputs.append(competitor_outputs[i][sent_ids])

    return shuffle_records, shuffle_outputs


def prepare_for_he():
    input_file = "HE/input.txt"
    ofnames = ["HE/vae.txt", "HE/dss-vae.txt", "HE/syntax-tvae.txt"]

    inputs, competitor_outputs = combine_outputs(input_file, *ofnames)

    records, outputs = record_the_shuffle(inputs, competitor_outputs)

    write_docs("./HE/record.txt", records)
    write_docs("./HE/output.txt", records)


def combine_the_record(record_file, *score_file):
    record_ids = load_docs(record_file)
    record_ids = [int(i) for i in record_ids]
    scores = [0 for _ in range(max(record_ids) + 1)]

    for sfile in score_file:
        score_output = load_docs(sfile)
        score_output = [int(i) for i in score_output]
        for s, r in zip(score_output, record_ids):
            scores[r] += s

    print(scores)


def get_the_record(record_file, *score_file):
    model_ids = load_docs(record_file)
    model_ids = [int(i) for i in model_ids]
    model_num = max(model_ids) + 1
    eval_num = len(score_file)
    # scores = [[] for _ in range(max(model_ids) + 1)]
    # prepare for load score
    raw_score = []
    human_scores = [[[] for _ in range(model_num)] for _ in range(len(score_file))]
    for hum, file in enumerate(score_file):
        score_output = load_docs(file)
        score_output = [int(i) for i in score_output]
        raw_score.append(score_output)
        for s, m in zip(score_output, model_ids):
            # scores[m].append(s)
            human_scores[hum][m].append(s)

    sample_num = len(human_scores[0][0])

    print("model_num:", model_num)
    print("human_num:", eval_num)
    print("sample_num:", sample_num)

    import numpy as np

    human_model_score = [
        [np.mean(s) for s in human_score] for human_score in human_scores
    ]
    print(human_model_score)

    model_human_score = [
        [human_model_score[i][j] for i in range(eval_num)] for j in range(model_num)
    ]
    print(model_human_score)

    mean = [np.mean(m_h) for m_h in model_human_score]
    var = [np.var(m_h) for m_h in model_human_score]

    print("mean:", mean)
    print("var:", var)

    # print(human_scores)
    # human_metrics = []

    # from sklearn import metrics
    # agree_score = [[0 for _ in range(len(raw))] for _ in range(len(raw))]
    # for i in range(len(raw)):
    #     for j in range(len(raw)):
    #         agree_score[i][j] = metrics.cohen_kappa_score(raw[i], raw[j])
    # print("agree:", agree_score)
    # score = [sum(s) for s in scores]
    # final_score = [s / 300 for s in score]
    # print(final_score)


if __name__ == "__main__":
    fluency_score_files = ["HE/fluency_baoy.txt", "HE/fluency_taok.txt", "HE/fluency_wangr.txt"]
    relevance_score_files = ["HE/relevance_baoy.txt", "HE/relevance_taok.txt", "HE/relevance_wangr.txt"]

    # combine_the_record("HE/record.txt", *fluency_score_files)
    # combine_the_record("HE/record.txt", *relevance_score_files)

    get_the_record("HE/record.txt", *fluency_score_files)
    get_the_record("HE/record.txt", *relevance_score_files)
