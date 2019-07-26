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
PAD_ID = 1

def _padding(tokens_list, max_len):
    ret = np.zeros((len(tokens_list),max_len),np.int32)
    for i,t in enumerate(tokens_list):
        t = t + (max_len-len(t)) * [PAD_ID]
        ret[i] = t
    return ret

def convert_tokens_to_id(tokens, w2i, max_tokens=1200):
    ids = [w2i.get(t,UNK_ID) for t in tokens]
    if max_tokens != -1:
        ids = ids[:max_tokens]
    return ids

class DataItem(namedtuple("DataItem",('tokens','length','labels','raw_tokens'))):
    pass

class DataSet(object):
    def __init__(self, data_files, vocab_file, label_file, batch_size=32, max_len = 1200, mode='train'):
        self.data_files = data_files
        self.batch_size = batch_size
        self.max_len = max_len
        self.mode = mode

        self.vocab, self.w2i = read_vocab(vocab_file)
        self.i2w = {v:k for k,v in self.w2i.items()}
        self.label_names, self.l2i = read_vocab(label_file)
        self.i2l = {v:k for k,v in self.l2i.items()}

        self._raw_data = []
        self._preprocess()


    def _preprocess(self):
        print_out("# Start to preprocessing data...")
        for fname in self.data_files:
          print_out("# load data from %s ..." % fname)
          for i,line in enumerate(open(fname)):
            if not line.strip():
              continue
            labels = ['O','O']
            tokens = ['[CLS]','[SEP]']
            if self.mode != 'inference':
              segments = line.strip().split()
              for segment in segments:
                  seg_tokens,seg_label = segment.split('/')
                  seg_tokens = seg_tokens.split('_')
                  tokens.extend(seg_tokens)
                  if seg_label == 'o':
                    labels.extend(['O']*len(seg_tokens))
                  else:
                    seg_labels = ['I-{}'.format(seg_label)] * len(seg_tokens)
                    seg_labels[0] = 'B-{}'.format(seg_label)
                    labels.extend(seg_labels) 
            else:
              tokens.extend(line.strip().split('_'))
              labels = ['O'] * len(tokens)
            assert len(tokens) == len(labels), "tokens length not equal to label: {}".format(line)
            tokens_id = convert_tokens_to_id(tokens, self.w2i, self.max_len)
            labels = convert_tokens_to_id(labels,self.l2i,self.max_len)

            self._raw_data.append(DataItem(raw_tokens=tokens,tokens=tokens_id,labels=labels,length=len(tokens)))
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
        contents = [item.tokens for item in batch]
        targets = [item.labels for item in batch]
        lengths = [item.length for item in batch]
        contents = _padding(contents,max(lengths))
        targets = _padding(targets,max(lengths))
        return np.asarray(contents), np.asarray(lengths), np.asarray(targets)

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
