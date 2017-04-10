#!usr/bin/env python3
#-*- coding:UTF-8 -*-

import util
import directories
import data
import TextProcessor
import model

def run(path):
    dataset_name = util.dataset_name(path)
    doc = data.Document(path)
    sentences = TextProcessor.get_allsentarray(doc)
    # mention
    mentions = util.set_mentions(dataset_name, doc)
    #段落分け
    sep_mentions = TextProcessor.mention_separater(mentions)
    #モデルに段落分けしたmentionを入力
    model.main(sep_mentions, sentences)


if __name__ == '__main__':
    print('making train dataset.')
    run(directories.TRAIN_PATH)
    print('finished.')
    # print('making dev dataset.')
    # run(directories.DEV_PATH)
    # print('finished.')
    # print('making test dataset.')
    # run(directories.TEST_PATH)
    # print('finished.')