#!usr/bin/env python3
#-*- coding:UTF-8 -*-

import ipdb
import tensorflow as tf
import numpy as np
import random
import time
import collections

import wordvector
import directories
import TextProcessor

BATCH_SIZE = 200
WINDOW_SIZE = 4

def build_vocab(sents):
    words = []
    for word in sents:
        words.append(word)
    counter = collections.Counter(words)
    count_pair = sorted(counter.items(), key = lambda x: (-x[1], x[0]))
    wordList, _ = list(zip(*count_pair))
    word_to_id = dict(zip(wordList, range(len(wordList))))
    word_to_id["<unk>"] = len(wordList) + 1
    return word_to_id

def get_lookuptable(word_to_id, vector):
    lookuptable = []
    for word, _ in sorted(word_to_id.items(), key=lambda x: x[1]):
        lookuptable.append(vector.get(word))
    return np.array(lookuptable,dtype=np.float32)

def placeholder_inputs():
    with tf.name_scope(u'place_holder') as scope:
        input_a = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
        input_m = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
        feature_a = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None])
        feature_m = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None])
        label = tf.placeholder(tf.float32, shape=[None, ])

    return input_a, input_m, label


def mention_encoder(x):
    # hidden1
    with tf.name_scope(u'hidden_layer1') as scope:
        weight1 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([200, 100], stddev=0.2, dtype=tf.float32), name='weight'), 0)
        bias1 = tf.Variable(tf.zeros([100]), name='bias')
        hidden1 = tf.matmul(tf.reshape(x,[200,200]), weight1) + bias1

    # hidden2
    with tf.name_scope(u'hidden_layer2') as scope:
        weight2 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([100, 100], stddev=0.2), name='weight'), 0)
        bias2 = tf.Variable(tf.zeros([100]), name='bias')
        hidden2 = tf.matmul(hidden1, weight2) + bias2

    # representation
    with tf.name_scope(u'representaton') as scope:
        weight3 = tf.nn.l2_normalize(tf.Variable(tf.truncated_normal([100, 50], stddev=0.2), name='weight'), 0)
        bias3 = tf.Variable(tf.zeros([50]), name='bias')
        rep = tf.matmul(hidden2, weight3) + bias3

    return rep

def scoring_func(m, a):
    with tf.name_scope(u'scoring') as scope:
        bias_p = tf.Variable(tf.zeros([BATCH_SIZE]), name='bias')
        score = tf.reduce_sum(tf.mul(m, a)) + bias_p

    return score

#コサイン類似度の計算
# def culc_simcos(a, m):
#     with tf.name_scope(u'sim_cos') as scope:
#         a_dot_m = tf.matmul(a, m, transpose_b=True)
#         a_abs = tf.abs(a)
#         m_abs = tf.abs(m)
#         cos = tf.div(a_dot_m, tf.mul(a_abs, m_abs))
#
#     return cos

def loss_func(score, y):
    return -tf.reduce_sum(y * tf.log(score) + (1 - y) * tf.log(1 - score))

def fill_feed_dict(inputs, vectors, ph_a, ph_m, l, word_to_id):
    while True:
        random.seed(time.time())
        rand = random.randint(0, len(inputs) - 1)
        batch = inputs[rand]
        if len(batch) < 100:
            break

    m_vectors, a_vectors, labels = TextProcessor.pair_maker(batch, word_to_id)

    feed_dict = {
        ph_a : a_vectors,
        ph_m : m_vectors,
        l    : labels
    }

    return feed_dict

def main(inputs, sentences):
    epoch = 300
    word_to_id = build_vocab(sentences)
    vectors = wordvector.WordVector()

    graph = tf.Graph()
    with graph.as_default():

        # placeholderの用意
        input_a, input_m, label = placeholder_inputs()

        # lookuptableの作成
        lookuptabel = tf.Variable(initial_value=get_lookuptable(word_to_id, vectors))
        a_embed = tf.nn.embedding_lookup(lookuptabel, input_a)
        m_embed = tf.nn.embedding_lookup(lookuptabel, input_m)

        # 先行詞とターゲットのNN
        rep_a = mention_encoder(a_embed)
        rep_m = mention_encoder(m_embed)

        # コサイン距離計算
        cos = scoring_func(rep_m, rep_a)
        #cos = culc_simcos(rep_a, rep_m)

        # 損失関数
        loss = loss_func(cos, label)

        # 正答率
        corecct_prediction = tf.equal(tf.round(cos),tf.round(label))
        acc = tf.reduce_mean(tf.cast(corecct_prediction,tf.float32))
        test_acc = tf.reduce_mean(tf.cast(corecct_prediction,tf.float32))
        loss_sum = tf.scalar_summary(loss.op.name,loss)
        acc_sum = tf.scalar_summary('accuracy',acc)
        test_acc_sum = tf.scalar_summary('test_accuracy',test_acc)

        train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        summary_op = tf.merge_all_summaries()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.initialize_all_variables()
        sess.run(init)
        summary_writer = tf.train.SummaryWriter(directories.LOG_DIR,graph_def=sess.graph_def)
        start_train = time.time()
        for i in range(1000):
            feed_dict = fill_feed_dict(inputs, vectors, input_a, input_m, label, word_to_id)
            _, l = sess.run([train_step, loss], feed_dict=feed_dict)
            ipdb.set_trace()
            if (i < 10 and i != 0):
                print(str(i) + ' epoch',end=' ')
                print('loss:' + str(l))
                summary_str = sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,i)
            elif(i % 10 == 0 and i != 0):
                print(str(i) + ' epoch',end=' ')
                print('loss:' + str(l))
                summary_str = sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,i)
        print('-----training finish-----')
