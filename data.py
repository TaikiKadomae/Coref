#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from bs4 import BeautifulSoup as bs
import pickle
import directories


class Mention():
    def __init__(self, surface, mention_id, doc_id, sent, dep, tag):

        self.doc_id = doc_id
        self.mention_id = mention_id
        self.mention_num = mention_id
        self.sent_num = sent.sent_num
        self.sent_id = sent.sent_id
        self.start_index = sent.get_word_position(surface[0]) - 1
        self.end_index = sent.get_word_position(surface[-1])
        self.head_index = sent.get_word_position(surface[-1])
        self.mention_type = surface[0].check_type()
        self.dep_parent, self.dep_relation = dep
        self.sentence = sent.get_sent_array()
        self.contained_in_other_mention = 0
        self.mention_pair = tag.get_mention_pair()
        self.tag_id = tag.tag_id
        self.words = surface
        def join_surface():
            join = ''
            for w in self.words:
                join += w.surface
            return join
        self.surface = join_surface()

class MentionFeature():
    def __init__(self, head_match, exact_match, relaxed_match):
        self.same_speaker = 0
        self.antecedent_is_mention_speaker = 0
        self.mention_is_antecedent_speaker = 0
        self.relaxed_head_match = head_match
        self.exact_string_match = exact_match
        self.relaxed_string_match = relaxed_match

class Word():
    def __init__(self, surface, kana, origin,
                 pos, dpos, conj, conj_form, position,
                 tag_id, phrase_id):

        self.surface = surface
        self.kana = kana
        self.origin = origin
        self.pos = pos
        self.dpos = dpos
        self.conj = conj
        self.conj_form = conj_form
        self.position = position
        self.tag_id = tag_id
        self.phrase_id = phrase_id

    def check_type(self):
        if self.dpos == '固有名詞':
            return 'PROPER'

        elif self.dpos in ['名詞形態指示詞', '連体詞形態指示詞']:
            return 'PRONOMINAL'

        else:
            return 'NOMINAL'


class Tag():
    def __init__(self, tags, tag_id, phrase_id, dep, rel):
        self.tag_id = tag_id
        self.phrase_id = phrase_id
        self.dep = dep.strip()[0:-1]
        self.dep_type = dep.strip()[-1]
        self.rel = rel

        def set_words(ts):
            words = []
            i = 1
            for l in ts:
                if l.startswith(('+', '*')):
                    continue
                w = l.split()
                words.append(Word(w[0], w[1], w[2], w[3], w[4], w[5], w[6], i, self.tag_id, phrase_id))
                i = i + 1
            return words

        self.words = set_words(tags)
        self.word_num = len(self.words)

    def get_next_word(self, word):
        return self.words[word.position + 1]

    def word_is_end(self, word):
        return word.position == self.word_num

    def get_dep(self, tag):
        soup = bs(self.rel, 'lxml')
        for s in soup.find_all('rel'):
            if tag.tag_id == s.get('tag') and s.get('type') not in ['=', '≒']:
                return self.words[0].surface, s.get('type')
        return self.words[0].surface, 'no-parent'

    def get_mention_pair(self):
        soup = bs(self.rel, 'lxml')
        ret =[]
        for s in soup.find_all('rel'):
            if s.get('type') in ['=','  ≒']:
                ret.append([s.get('sid'),s.get('tag')])

        return ret

    def get_target(self):
        soup = bs(self.rel, 'lxml')
        ret = []
        for s in soup.find_all('rel'):
            if s.get('type') in ['=', '≒']:
                ret.append(s.get('target'))
        return ret

class Phrase():
    def __init__(self, phrases, phrase_id):
        self.phrase_id = phrase_id

        def set_Tags(ps):
            line = []
            ts = []
            t_id = []
            dep = []
            rel = []
            first = True
            for l in ps:
                spl = l.split(' ', 3)
                if l.startswith('+'):
                    t_id.append(spl[1])
                    dep.append(spl[2])
                    if len(spl) > 3:
                        rel.append(spl[3])
                    else:
                        rel.append('None')

                    if not first:
                        ts.append(line)
                        line = []
                    else:
                        first = False
                line.append(l)
            else:
                ts.append(line)

            ret = []
            for (t, id, d, r) in zip(ts, t_id, dep, rel):
                ret.append(Tag(t, id, self.phrase_id, d, r))

            return ret

        self.tags = set_Tags(phrases)

    def get_word_num(self):
        total = 0
        for t in self.tags:
            total = total + t.word_num
        return total


class Sentence():
    def __init__(self, sentences, sent_id, sent_num):
        self.sent_num = sent_num
        self.sent_id = sent_id
        self.sentences = sentences
        def set_phrases(sents):
            line = []
            phras = []
            p_id = []
            first = True
            for l in sents:
                if l.startswith('*'):
                    p_id.append(l.split()[1])
                    if not first:
                        phras.append(line)
                        line = []
                    else:
                        first = False
                line.append(l)
            else:
                phras.append(line)

            ret = []
            for (p, id) in zip(phras, p_id):
                ret.append(Phrase(p, id))
            return ret
        self.phrases = set_phrases(sentences)

    def get_sent_num(self):
        return self.sent_id

    def get_sent_array(self):
        sent = []
        for line in self.sentences:
            if line.split()[0] not in ["+", "*", "#","EOS"]:
                sent.append(line.split()[0])
            else:
                continue
        return sent

    def get_word_position(self, word):
        position = 0
        for p in self.phrases:
            if p.phrase_id == word.phrase_id:
                for t in p.tags:
                    if t.tag_id == word.tag_id:
                        position = position + word.position
                        return position - 1
                    else:
                        position = position + t.word_num
            else:
                position = position + p.get_word_num()

    def get_dep(self, tag):
        if tag.dep == '-1':
            return 'no-parent', '<missing>'
        for p in self.phrases:
            for t in p.tags:
                if t.tag_id == tag.dep:
                    w, ty = t.get_dep(tag)
                    return w, ty


class Paragraph():
    def __init__(self, paragraphs, paragraph_id):
        self.paragraphs = paragraphs
        self.paragraph_id = paragraph_id


        def set_sentences(paragraphs):
            line = []
            s_id = []
            s_num = []
            sents = []
            i = 0
            for l in paragraphs:
                line.append(l)
                if l.startswith('#'):
                    s_id.append(l.split()[1].split(':')[1])
                if l.strip() == 'EOS':
                    line.pop()
                    line.pop(0)
                    sents.append(line)
                    s_num.append(i)
                    i = i + 1
                    line = []

            ret = []
            for (s, id, num) in zip(sents, s_id, s_num):
                ret.append(Sentence(s, id, num))

            return ret

        self.sentences = set_sentences(self.paragraphs)

    def get_sentences(self):
        line = []
        sents = []
        for l in self.paragraphs:
            if not l.split()[0] in ['+', '*', '#', 'EOS']:
                line.append(l.split()[0])
            if l.strip() == 'EOS':
                sents.append(line)
                line = []

        return sents


class Document():
    def __init__(self, file_path):

        self.file_path = file_path

        def set_paragraph(f_path):
            line = []
            pa_id = []
            paras = []
            curr_id = 'in'
            prev_id = 'ini'
            isFirst = True
            i = 0
            for l in open(f_path):
                line.append(l)
                if l.startswith('#'):
                    curr_id = l.split()[1].split(':')[1].split('-')[0]
                    if curr_id != prev_id and not isFirst:
                        line.pop()
                        paras.append(line)
                        pa_id.append(i)
                        i = i + 1
                        line = []
                        line.append(l)
                    prev_id = curr_id
                isFirst = False
            paras.append(line)
            pa_id.append(i)
            ret = []
            for (pa, paid) in zip(paras, pa_id):
                ret.append(Paragraph(pa, paid))

            return ret, paras

        self.paragraphs, self.raw = set_paragraph(file_path)
        # with open(directories.PICKLE_PATH + 'paragraphs.pickle', 'wb') as f:
        #     pickle.dump(self.raw,f)

    def get_doc_id(self):
        return self.file_path.split('/')[-1].split('.')[0]

    def count_word(self):
        i = 0
        for pa in self.paragraphs:
            for s in pa.sentences:
                for p in s.phrases:
                    for t in p.tags:
                        for w in t.words:
                            i = i + 1
        return i
