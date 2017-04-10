# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import numpy as np
import data
import copy
import time
import directories
import wordvector
import ipdb

def get_allsentarray(doc):
    sents = []
    for para in doc.paragraphs:
        sents.extend(para.get_sentences())

    return [inner for outer in sents for inner in outer]

def get_vec(m, word_to_id):

    ret = []
    window = 2
    forward = m.start_index - 1
    backward = m.end_index
    f_deadend = False
    forward_vec = []

    for i in range(window):

        if not forward - i <= -1:
            f_id = m.start_index - 1 - i
            f_word = m.sentence[f_id]
            f_vec = word_to_id[f_word]
            forward_vec.append(f_vec)
        else:
            f_deadend = True

        if f_deadend:
            #forward_vec.insert(0, np.zeros(directories.VEC_SIZE, dtype='float32'))
            #forward_vec[0:0] = np.zeros(directories.VEC_SIZE, dtype='float32')
            forward_vec.append(word_to_id["<unk>"])
    backward_vec = []
    b_deadend = False
    for k in range(window):

        if not backward + k >= len(m.sentence):
            b_id = m.end_index + k
            b_word = m.sentence[b_id]
            b_vec = word_to_id[b_word]
            backward_vec.append(b_vec)
        else:
            b_deadend = True

        if b_deadend:
            # backward_vec.append(np.zeros(directories.VEC_SIZE, dtype='float32'))
            #backward_vec[-1:-1] = np.zeros(directories.VEC_SIZE, dtype='float32')
            backward_vec.append(word_to_id["<unk>"])
    ret = forward_vec + backward_vec
    return ret

def mention_separater(mentions):
    print('paragraph loading...')
    time.sleep(0.01)
    total = len(mentions)
    PB = tqdm(total=total)
    para = []
    ret = []
    id = 0
    mention_id = 0
    for m in mentions:
        if m.doc_id == id:
            para.append(m)
            m.mention_id = mention_id
            m.mention_num = mention_id
            mention_id = mention_id + 1
        else:
            ret.append(para)
            para = []
            mention_id = 0
            m.mention_id = mention_id
            m.mention_num = mention_id
            para.append(m)
            id = m.doc_id
        PB.update(1)
    ret.append(para)
    PB.close()
    return ret

def mentionExtractor(doc):
    total = doc.count_word()
    PB = tqdm(total=total)
    Mentions = []
    # mentionscluster = []
    mention_id = 0

    for para in doc.paragraphs:
        for sent in para.sentences:
            for phra in sent.phrases:
                for tag in phra.tags:
                    prev_surface = 'ini'
                    prev_pos = 'ini'
                    prev_dpos = 'ini'
                    surface = []
                    for word in tag.words:
                        PB.update(1)
                        if word.pos == '名詞':
                            surface.append(word)

                        elif word.pos == '形容詞':
                           surface.append(word)

                        elif word.dpos == '名詞性述語接尾辞':
                            if prev_pos == '形容詞':
                               surface.append(word)
                            else:
                                surface = []
                                continue
                        elif word.dpos == '名詞性名詞助数辞' and prev_dpos == '数詞':
                            surface.append(word)

                        elif word.dpos in ['名詞形態指示詞', '名詞性名詞接尾辞', '連体詞形態指示詞','名詞接頭辞']\
                                and not prev_dpos == '人名':
                            surface.append(word)

                        elif word.dpos == '名詞性特殊接尾辞' and prev_dpos == '地名':
                            surface.append(word)

                        elif word.surface == '・' and prev_dpos == '人名':
                            surface.append(word)

                        if tag.word_is_end(word) and len(surface) > 0:
                            Mentions.append(data.Mention(surface, mention_id, para.paragraph_id,
                                                         sent, sent.get_dep(tag), tag))
                            mention_id = mention_id + 1

                        prev_surface = word.surface
                        prev_pos = word.pos
                        prev_dpos = word.dpos
     #   mentionscluster.append(Mentions)
    PB.close()
    time.sleep(0.1)
    return Mentions

def labelMaker(doc, mentions):
    time.sleep(0.01)
    m_para = mentions
    p_para = copy.deepcopy(mentions)
    print("feature and label extracting...")
    time.sleep(0.01)
    total = len(mentions)
    PB = tqdm(total=total)
    paralabel = []
    parafeature = []
    gold = []
    goldpair = []
    isfound = False
    count = 0

    for ms, ps in zip(m_para, p_para):
        label = []
        feature = []
        for m in ms:
            for p in ps:
                head_match = 0
                exact_match = 0
                relaxed_match = 0
                if not m.mention_id == p.mention_id:
                    # 共参照関係抽出
                    for mp in m.mention_pair:
                        if mp[0] == p.sent_id and mp[1] == p.tag_id:
                            label.append((str(m.mention_id) + ' ' + str(p.mention_id), 1))
                            isfound = True
                            count = count + 1
                            goldpair.append([m.mention_id, p.mention_id])
                            break
                    for pp in p.mention_pair:
                        if pp[0] == m.sent_id and pp[1] == m.tag_id and not isfound:
                            label.append((str(m.mention_id) + ' ' + str(p.mention_id), 1))
                            isfound = True
                            goldpair.append([m.mention_id, p.mention_id])
                            count = count + 1
                            break
                    if not isfound:
                        label.append((str(m.mention_id) + ' ' + str(p.mention_id), 0))
                    isfound = False

                    # 特徴抽出
                    if m.words[-1] in p.words:
                        head_match = 1
                    if m.surface == p.surface:
                        exact_match = 1
                        head_match = 1
                        relaxed_match = 1
                    if m.surface in p.surface:
                        relaxed_match = 1
                    feature.append((str(m.mention_id) + ' ' + str(p.mention_id),
                                    data.MentionFeature(head_match, exact_match, relaxed_match)))
                    if head_match or exact_match or relaxed_match:
                        f = open(directories.STRING_MATCH + 'match.txt', 'a')
                        f.write("{}, {}  {} {} {} \n".format(m.surface, p.surface, head_match, exact_match, relaxed_match))
                        f.close()
            ps.pop(0)
        gold.append(goldpair)
        goldpair = []
        paralabel.append(label)
        parafeature.append(feature)
        PB.update(1)
    time.sleep(0.01)
    PB.close()

    return paralabel, parafeature, gold

#共参照関係抽出
def get_coref(m, a):
    if not m.mention_id == a.mention_id:
        # 共参照関係抽出
        for mp in m.mention_pair:
            if mp[0] == a.sent_id and mp[1] == a.tag_id:

                return 1

        for pp in a.mention_pair:
            if pp[0] == m.sent_id and pp[1] == m.tag_id:

                return 1

    return 0

def goldMaker(golds):
    ret = []
    print('making gold data...')
    PB = tqdm(len(golds))
    for gold in golds:
        ret.append(recall(gold))
        PB.update(1)

    time.sleep(0.01)
    PB.close()
    return ret

def recall(gold):
    id = 0
    newgold = []
    change = False
    copygold = copy.deepcopy(gold)
    for pair in gold:
        tiny_change = False
        for cp in copygold:
            if not pair == cp:
                sets = set(pair) & set(cp)
                if not len(sets) == 0:
                    newgold.append(list(set(pair) | set(cp)))
                    tiny_change = True
                    change = True
                    copygold.pop(copygold.index(cp))
                    gold.remove(cp)
        id = id + 1
        if not tiny_change:
            newgold.append(pair)
    if change:
        return recall(newgold)
    else:
        return newgold

def pair_maker(mentions, word_to_vec):
    ret_m = []
    ret_a = []
    ret_l = []
    copy_mentions = copy.deepcopy(mentions)
    vecs = []
    for i, m in enumerate(mentions):
        for k, a in enumerate(copy_mentions):
            if not i == k:
                m_vec = get_vec(m, word_to_vec)
                a_vec = get_vec(a, word_to_vec)
                label = get_coref(m, a)
                ret_m.append(m_vec)
                ret_a.append(a_vec)
                ret_l.append(label)

    return ret_m, ret_a, ret_l
