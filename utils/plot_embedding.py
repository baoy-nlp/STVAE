# coding='utf-8'
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

PLOT_NUM = 5000
FILE_NUM = 5000


def get_data(word_file='../out/words_{}.txt'.format(FILE_NUM), syn_file='../out/syn_{}.txt'.format(FILE_NUM),
             sem_file='../out/sem_{}.txt'.format(FILE_NUM)):
    word_set = pd.read_csv(word_file, sep='\t')
    syn_set = pd.read_csv(syn_file, sep='\t')
    sem_set = pd.read_csv(sem_file, sep='\t')

    label = np.concatenate(
        [word_set.values[:PLOT_NUM], word_set.values[:PLOT_NUM]], axis=0)
    data = np.concatenate(
        [syn_set.values[:PLOT_NUM], sem_set.values[:PLOT_NUM]], axis=0)
    return data, label


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(i // PLOT_NUM), color=plt.cm.Set1(i // PLOT_NUM),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def write_to_json(data, label, file='../out/dss_vae_{}.json'.format(PLOT_NUM)):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    ret_list = []
    with open(file, 'w') as f:
        for i in range(data.shape[0]):
            ret_item = [data[i, 0].item(), data[i, 1].item(), str(label[i][0]), i // PLOT_NUM]
            ret_list.append(ret_item)
        json.dump(ret_list, f)
    print("write_finish")


def main():
    data, label = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='random', random_state=0,
                perplexity=43, learning_rate=10, n_iter=2000)
    result = tsne.fit_transform(data)
    write_to_json(result, label)
    # fig = plot_embedding(
    # result, label, 't-SNE embedding of the DSS-VAE')
    # plt.show(fig)


if __name__ == '__main__':
    main()
