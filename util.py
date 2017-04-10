#!usr/bin/env python3
#-*- coding:UTF-8 -*-

import directories
import data
import TextProcessor
import wordvector
import model

import ipdb
import time
import os
import random
import pickle


def dataset_name(path):
    if path == directories.TRAIN_PATH:
        return 'train'
    elif path == directories.DEV_PATH:
        return 'dev'
    elif path == directories.TEST_PATH:
        return 'test'

def set_mentions(dataset_name, doc):
    if not os.path.exists(directories.PICKLE_PATH + dataset_name + '_mention.pickle'):
        mentions = TextProcessor.mentionExtractor(doc)
        with open(directories.PICKLE_PATH + dataset_name + '_mention.pickle', 'wb') as f:
            pickle.dump(mentions,f)
        return mentions

    else:
        with open(directories.PICKLE_PATH + dataset_name + '_mention.pickle', 'rb') as f:
            return pickle.load(f)