#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 14:34
# @Author  : ZYQ
# @File    : main.py
# @Software: PyCharm

import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm


def regrx(text):
    text = re.sub(" 'm ", ' am ', text)
    text = re.sub(" 're ", ' are ', text)
    text = re.sub(" 'll ", ' will ', text)
    text = re.sub(" 've ", ' have ', text)
    text = re.sub(" n't ", ' not ', text)
    text = re.sub("[Ii]t 's", ' it is ', text)
    text = re.sub("[Cc]a ", ' can ', text)
    text = re.sub('\.{2,}', ' . ', text)
    text = re.sub(',{2,}', ' , ', text)
    text = re.sub('!{2,}', ' ! ', text)
    text = re.sub('\?{2,}', ' ? ', text)
    text = re.sub('\+{2,}', ' + ', text)
    text = re.sub('\*{2,}', ' * ', text)
    text = re.sub('#{2,}', ' # ', text)
    text = re.sub('-{2,}', ' - ', text)
    text = re.sub('%{2,}', ' % ', text)
    text = re.sub('~{2,}', ' ~ ', text)
    text = re.sub('@{2,}', ' @ ', text)
    text = re.sub('\${2,}', ' $ ', text)
    text = re.sub('\^{2,}', ' ^ ', text)
    text = re.sub('&{2,}', ' & ', text)
    text = re.sub('"', ' " ', text)
    text = re.sub('\(', ' ( ', text)
    text = re.sub('\)', ' ) ', text)
    text = re.sub('-', ' - ', text)
    text = re.sub('/', ' / ', text)

    return text


def load_vocab():
    vocab = set()
    domains = ['rest14', 'lap14']
    datasets = ['train', 'test']
    for domain in domains:
        for dataset in datasets:
            fin = open('./semeval14/{}_{}.raw'.format(domain, dataset), 'r')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = lines[i].strip().partition('$T$')
                aspect = lines[i + 1].strip()
                text = text_left + aspect + text_right
                text = regrx(text)
                vocab.update(text.lower().strip().split())
    return sorted(vocab)


def load_lexicon():
    lexicon = {}
    fin = open('lexicon/senticnet.txt', 'r', encoding='utf-8')
    for line in fin.readlines():
        token = line.strip().split()
        word, score = token[0], float(token[1])
        if '_' in word:
            continue
        elif -0.5 < score < 0.5:
            continue
        else:
            lexicon[word] = np.asarray(score, dtype='float32')
    return lexicon


def load_glove(vocab):
    fin = open('glove/glove.840B.300d.txt', 'r', encoding='utf-8')
    fout = open('glove/glove.semeval2014.txt', 'w', encoding='utf-8')
    lines = fin.readlines()
    for line in tqdm(lines):
        tokens = line.strip().split()
        word = ' '.join(tokens[:-300])
        if word in vocab:
            vocab.remove(word)
            fout.write(line)


def calculate_distance():
    fin = open('glove/glove.semeval2014.txt', 'r', encoding='utf-8')
    fout = open('knn/knn.pickle', 'wb')
    lines = fin.readlines()
    vocab = []
    embedding = {}
    for line in lines:
        tokens = line.strip().split()
        word, vec = ' '.join(tokens[:-300]), tokens[-300:]
        embedding[word] = np.asarray(vec, dtype='float32')
        vocab.append(word)
    data = {}
    matrix = np.zeros((len(vocab), len(vocab))).astype('float32')
    for i in tqdm(range(0, len(vocab))):
        matrix[i][i] = 0.
        for j in range(i + 1, len(vocab)):
            matrix[i][j] = np.linalg.norm(embedding[vocab[i]] - embedding[vocab[j]])
            matrix[j][i] = np.linalg.norm(embedding[vocab[i]] - embedding[vocab[j]])
    data['words'] = vocab
    data['dis'] = matrix
    pickle.dump(data, fout)
    fout.close()
    return


def find_top_k():
    fin = open('./knn/knn.pickle', 'rb')
    data = pickle.load(fin)
    fin.close()
    fout = open('knn/topK.pickle', 'wb')
    new_vocab = data['words']
    distance = data['dis']
    knn_dict = {}

    for i in tqdm(range(len(new_vocab))):
        dis_dict = {}
        for j in range(len(new_vocab)):
            dis_dict[new_vocab[j]] = distance[i][j]
        knn_dict[new_vocab[i]] = sorted(dis_dict.items(), key=lambda x: x[1], reverse=False)[1:11]
    pickle.dump(knn_dict, fout)
    fout.close()
    return


def similarity_ranking():
    fin = open('./knn/topK.pickle', 'rb')
    fin1 = open('./lexicon/senticnet.txt', 'r', encoding='utf-8')
    fout = open('./knn/topK_simi_score.pickle', 'wb')
    data = pickle.load(fin)
    lines = fin1.readlines()
    fin.close()
    fin1.close()
    score_dict = defaultdict(lambda: -255)
    for line in lines:
        word, score = line.strip().split()
        score_dict[word] = np.float32(score) + 1.
    word2score = {}
    for key, value in tqdm(data.items()):
        temp = {}
        for (word, simi) in value:
            temp[word] = abs(score_dict[key] - score_dict[word])
        word2score[key] = sorted(temp.items(), key=lambda x: x[1])
    pickle.dump(word2score, fout)
    fout.close()


def refine(alpha, beta):
    fin = open('./knn/topK_simi_score.pickle', 'rb')
    fin1 = open('glove/glove.semeval2014.txt', 'r', encoding='utf-8')
    fout = open('glove/re_glove.300d.txt', 'w', encoding='utf-8')
    data = pickle.load(fin)
    lines = fin1.readlines()
    fin.close()
    fin1.close()
    embedding = {}
    for line in lines:
        tokens = line.strip().split()
        word, vec = ' '.join(tokens[:-300]), tokens[-300:]
        embedding[word] = np.asarray(vec, dtype='float32')
    vocab = list(embedding.keys())

    V = np.zeros((len(vocab), 300)).astype('float32')  # (35739,300)
    for i in tqdm(range(len(vocab))):
        V[i, :] = embedding[vocab[i]]
    cond1 = os.path.exists('knn/weight.pickle')
    cond2 = os.path.exists('knn/similarity.pickle')
    if not cond1 or not cond2:
        fout2 = open('knn/weight.pickle', 'wb')
        fout3 = open('knn/similarity.pickle', 'wb')
        rank = [1 / i for i in range(1, 11)]
        W = np.zeros((len(vocab), len(vocab))).astype('float32')  # (35739,35739)
        S = np.zeros((len(vocab), len(vocab))).astype('float32')  # (35739,35739)
        for word, neighbor in tqdm(data.items()):
            for idx, (n, s) in enumerate(neighbor):
                W[vocab.index(word)][vocab.index(n)] = rank[idx]
                S[vocab.index(word)][vocab.index(n)] = s
        pickle.dump(W, fout2)
        pickle.dump(S, fout3)
        fout2.close()
        fout3.close()
    else:
        print('loading w and s')
        fin2 = open('knn/weight.pickle', 'rb')
        fin3 = open('knn/similarity.pickle', 'rb')
        W = pickle.load(fin2)
        S = pickle.load(fin3)
        fin2.close()
        fin3.close()

    v = V.T
    epoch = 0
    last_loss = 100
    start_time = datetime.now()
    loss_list = []

    while True:
        M = alpha * v + beta * np.matmul(v, W.T)  # (300,35739)
        B = alpha + beta * np.matmul(W, np.transpose(np.ones(len(vocab))))  # (35739,1)
        V = M / B  # (300,35739)

        a = np.sum(alpha * np.sqrt(np.sum((v - V) * (v - V), axis=0)))
        knn_dist = []
        for vec, weight in zip(V.T, W):
            V_t = np.tile(vec, (10, 1))
            temp = beta * np.sum(
                np.sqrt(np.sum((V_t - v.T[np.where(weight > 0)]) * (V_t - v.T[np.where(weight > 0)]), axis=1)))
            knn_dist.append(temp)
        b = np.sum(np.array(knn_dist), axis=0)
        loss = (a + b) / len(vocab)
        loss_list.append(loss)

        last = np.copy(v)
        now = np.copy(V)
        move = np.sum(np.sum(now.T - last.T, axis=1) / 300, axis=0) / len(vocab)

        # if epoch % 10 == 0:
        print('epoch:{} loss:{} move:{}'.format(epoch, round(loss, 6), abs(move)))
        v = V
        if last_loss - loss < 1:
            new_vec = v.T.tolist()
            for word, emb in zip(vocab, new_vec):
                assert len(emb) == 300
                emb = [str(i) for i in emb]
                vec = ' '.join(emb)
                fout.write(word + ' ' + vec + '\n')
            fout.close()
            end_time = datetime.now()
            print('epoch:{} loss:{} move:{}'.format(epoch, round(loss, 6), abs(move)))
            print('Train Time:{}'.format(end_time - start_time))
            fout4 = open('./loss.txt', 'w', encoding='utf-8')
            for i in loss_list:
                fout4.write(str(i) + '\n')
            break
        epoch += 1
        last_loss = loss


def eval(word):
    fin1 = open('./glove/glove.semeval2014.txt', 'r', encoding='utf-8')
    fin2 = open('./glove/re_glove.300d.txt', 'r', encoding='utf-8')
    lines1 = fin1.readlines()
    lines2 = fin2.readlines()
    fin1.close()
    fin2.close()
    vocab = []
    old_embedding, new_embedding = {}, {}
    for line1, line2 in zip(lines1, lines2):
        tokens1 = line1.strip().split()
        tokens2 = line2.strip().split()
        word1, vec1 = ' '.join(tokens1[:-300]), tokens1[-300:]
        word2, vec2 = ' '.join(tokens2[:-300]), tokens2[-300:]
        old_embedding[word1] = np.asarray(vec1, dtype='float32')
        new_embedding[word2] = np.asarray(vec2, dtype='float32')
        assert word1 == word2
        vocab.append(word1)

    old_dict, new_dict = {}, {}
    for i in tqdm(range(0, len(vocab))):
        old_dis = np.linalg.norm(old_embedding[word] - old_embedding[vocab[i]])
        new_dis = np.linalg.norm(new_embedding[word] - new_embedding[vocab[i]])
        old_dict[vocab[i]] = old_dis
        new_dict[vocab[i]] = new_dis
    old_top_k = sorted(old_dict.items(), key=lambda x: x[1])[1:11]
    new_top_k = sorted(new_dict.items(), key=lambda x: x[1])[1:11]
    print(old_top_k)
    print(new_top_k)

    tsne = TSNE(perplexity=3, metric='cosine')
    old = tsne.fit_transform([old_embedding[i[0]] for i in old_top_k] + [old_embedding[word]])
    new = tsne.fit_transform([new_embedding[i[0]] for i in new_top_k] + [new_embedding[word]])
    old_word_list = [i[0] for i in old_top_k] + [word]
    new_word_list = [i[0] for i in new_top_k] + [word]
    x1, y1 = old[:, 0], old[:, 1]
    x2, y2 = new[:, 0], new[:, 1]

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.scatter(x1, y1, c='b')
    for i in range(len(old)):
        plt.annotate(old_word_list[i], xy=(x1[i], y1[i]),
                     xytext=(x1[i] + 0.1, y1[i] + 0.1))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    plt.title('old')
    fig.add_subplot(1, 2, 2)
    plt.scatter(x2, y2, c='r')
    for i in range(len(new)):
        plt.annotate(new_word_list[i], xy=(x2[i], y2[i]),
                     xytext=(x2[i] + 0.1, y2[i] + 0.1))  # 这里xy是需要标记的坐标，xytext是对应的标签坐标
    plt.title('new')
    plt.savefig('./result/' + word)
    plt.show()


def main():
    process = False  # do not change unless you use another lexicon or embedding
    train = False  # you can choose alpha and beta to train new embeddings
    if process:
        lexicon = load_lexicon()
        lexicon_vocab = list(lexicon.keys())  # 36096
        load_glove(lexicon_vocab)  # 31442
        calculate_distance()
        find_top_k()
        similarity_ranking()
    if train:
        refine(alpha=1, beta=0.1)
    eval('recommend')
    eval('satisfied')
    eval('unlucky')


if __name__ == '__main__':
    main()
