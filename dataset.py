# ======================================== 
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================
import codecs
import json
from collections import namedtuple

import numpy as np
import tensorflow as tf

from utils import print_out
from thrid_utils import read_vocab

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list),max_len),np.int32)
    for i,t in enumerate(tokens_list):
        t = t + (max_len-len(t)) * [EOS_ID]
        ret[i] = t
    return ret

def _tokenize(content, w2i, max_tokens=1200, reverse=False, split=True):
    def get_tokens(content):
        tokens = content.strip().split()
        ids = []
        for t in tokens:
            if t in w2i:
                ids.append(w2i[t])
            else:
                for c in t:
                    ids.append(w2i.get(c,UNK_ID))
        return ids
    if split:
        ids = get_tokens(content)
    else:
        ids = [w2i.get(t,UNK_ID) for t in content.strip().split()]
    if reverse:
        ids = list(reversed(ids))
    tokens = [SOS_ID] + ids[:max_tokens] + [EOS_ID]
    return tokens

class DataItem(namedtuple("DataItem",('content','length','labels','id'))):
    pass

class DataSet(object):
    def __init__(self, data_files, vocab_file, label_file, batch_size=32, reverse=False, split_word=True, max_len = 1200):
        self.reverse = reverse
        self.split_word = split_word
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_len = max_len

        self.vocab, self.w2i = read_vocab(vocab_file)
        self.i2w = {v:k for k,v in self.w2i.items()}
        self.label_names, self.l2i = read_vocab(label_file)
        self.i2l = {v:k for k,v in self.l2i.items()}

        self.tag_l2i = {"1":0,"0":1,"-1":2,"-2":3}
        self.tag_i2l = {v:k for k,v in self.tag_l2i.items()}

        self._raw_data = []
        self.items = []
        self._preprocess()

    def get_label(self, labels, l2i, normalize=False):
        one_hot_labels = np.zeros(len(l2i),dtype=np.float32)
        for n in labels:
            if n:
                one_hot_labels[l2i[n]] = 1
        
        if normalize:
            one_hot_labels = one_hot_labels / len(labels)
        return one_hot_labels

    def _preprocess(self):
        print_out("# Start to preprocessing data...")
        for fname in self.data_files:
            print_out("# load data from %s ..." % fname)
            for line in open(fname):
                item = json.loads(line.strip())
                content = item['content']
                content = _tokenize(content, self.w2i, self.max_len, self.reverse, self.split_word)
                item_labels = []
                for label_name in self.label_names:
                    labels = [item[label_name]]
                    labels = self.get_label(labels,self.tag_l2i)
                    item_labels.append(labels)
                self._raw_data.append(DataItem(content=content,labels=np.asarray(item_labels),length=len(content),id=int(item['id'])))
                self.items.append(item)

        self.num_batches = len(self._raw_data) // self.batch_size
        self.data_size = len(self._raw_data)
        print_out("# Got %d data items with %d batches" % (self.data_size, self.num_batches))

    def _shuffle(self):
        # code from https://github.com/fastai/fastai/blob/3f2079f7bc07ef84a750f6417f68b7b9fdc9525a/fastai/text.py#L125
        idxs = np.random.permutation(self.data_size)
        sz = self.batch_size * 50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=lambda x:self._raw_data[x].length, reverse=True) for s in ck_idx])
        sz = self.batch_size
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self._raw_data[ck[0]].length for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

    def process_batch(self, batch):
        contents = [item.content for item in batch]
        lengths = [item.length for item in batch]
        contents = _padding(contents,max(lengths))
        lengths = np.asarray(lengths)
        targets = np.asarray([item.labels for item in batch])
        ids = [item.id for item in batch]
        return contents, lengths, targets, ids

    def get_next(self, shuffle=True):
        if shuffle:
            idxs = self._shuffle()
        else:
            idxs = range(self.data_size)

        batch = []
        for i in idxs:
            item = self._raw_data[i]
            if len(batch) >= self.batch_size:
                yield self.process_batch(batch)
                batch = [item]
            else:
                batch.append(item)
        if len(batch) > 0:
            yield self.process_batch(batch)
