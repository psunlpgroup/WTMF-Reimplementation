
from __future__ import print_function

import os
import sys
import math
import time
import pickle
from collections import defaultdict

indir = sys.argv[1]
outdir = sys.argv[2]
min_word_count = int(sys.argv[3])

if not os.path.isdir(indir):
    print('Error: ' + indir + ' is not a directory!')
    exit()

if not os.path.isdir(outdir):
    print('Error: ' + outdir + ' is not a directory!')
    exit()

if not (min_word_count >= 1):
    print('Error: min word count must be >= 1')
    exit()


def return0():
    return 0

def is_valid_file(fpath):
    return os.path.isfile(fpath) and os.path.basename(fpath)[0] != '.'

def read_chunks(indir):
    if not os.path.isdir(indir):
        print('Error: ' + indir + ' is not a directory!')
        return
    fpaths = [os.path.join(indir, f) for f in os.listdir(indir)]
    fpaths = filter(is_valid_file, fpaths)
    for (chunk_i, fpath) in enumerate(fpaths, 1):
        with open(fpath, 'r') as fin:
            for line in fin:
                yield (line, chunk_i)

def dump_data(data, fnames, outdirs):
    for (datum, fname, outdir) in zip(data, fnames, outdirs):
        fpath = os.path.join(outdir, fname)
        with open(fpath, 'w') as fout:
            pickle.dump(datum, fout)

def build_vocab(indir, min_word_count):
    vocab = []
    vocab_counts = {}
    for (line, _) in read_chunks(indir):
        tokens = line.split()
        for token in tokens:
            if token not in vocab_counts:
                vocab.append(token)
                vocab_counts[token] = 1;
            else:
                vocab_counts[token] += 1;
    vocab = filter(lambda x: vocab_counts[x] >= min_word_count, vocab)
    return vocab

def add_vocab(word, doc):
    index = word_to_index[word]
    index_counts[index] += 1
    if index not in docs[doc]:
        index_doc_counts[index] += 1
    docs[doc][index] += 1
    doc_counts[doc] += 1

def get_docs_and_counts(outdir):
    docs_dir = os.path.join(outdir, 'docs')
    counts_dir = os.path.join(outdir, 'counts')
    docs_paths = [os.path.join(docs_dir, fname) for fname in os.listdir(docs_dir)]
    counts_paths = [os.path.join(counts_dir, fname) for fname in os.listdir(docs_dir)]
    for (d_path, c_path) in zip(docs_paths, counts_paths):
        docs = pickle.load(open(d_path, 'r'))
        counts = pickle.load(open(c_path, 'r'))
        for (doc, count) in zip(docs, counts):
            yield (doc, count)

print('Setup... ', end='')

docs_dir = os.path.join(outdir, 'docs')
counts_dir = os.path.join(outdir, 'counts')
if not os.path.isdir(docs_dir):
    os.mkdir(docs_dir)
if not os.path.isdir(counts_dir):
    os.mkdir(counts_dir)

print('Done!')

print('Building vocabulary... ', end='')

vocab = build_vocab(infile, min_word_count)
word_to_index = {}
for word in vocab:
    word_to_index[word] = len(word_to_index)

print('Done!')

print('Loading corpus... ')

docs = []
doc_counts = []
index_counts = defaultdict(return0)
index_doc_counts = defaultdict(return0)


doc_n = 0
doc_chunk_n = 0
time_start = time.time()
chunk_current = 1
for (line, chunk_i) in read_chunks(indir):
    if chunk_i > chunk_current:
        dump_data(  (docs, counts),
                    (str(chunk_current), str(chunk_current)),
                    (docs_dir, counts_dir) )
        docs = []
        doc_counts = []
        doc_chunk_n = 0
        time_end = time.time()
        time_diff = time_end - time_start
        print('Doc {}: {}s (cumulative)'.format(chunk_current, round(time_diff, 2)))
        chunk_current = chunk_i
    docs.append(defaultdict(return0))
    doc_counts.append(0)
    tokens = line.split()
    for token in tokens:
        if token in word_to_index:
            add_vocab(token, doc_chunk_n)
    doc_n += 1
    doc_chunk_n += 1
dump_data(  (docs, doc_counts),
            (str(chunk_current), str(chunk_current)),
            (docs_dir, counts_dir) )
docs = []
doc_counts = []
time_end = time.time()
time_diff = time_end - time_start
print('Doc {}: {}s (cumulative)'.format(chunk_current, round(time_diff, 2)))

print('Done!')

print('Calculating tf-idf scores... ', end='')

index_idf = {index: math.log(float(doc_n) / (doc_count+1)) for (index, doc_count) in index_doc_counts.iteritems()}

doc_tfidf = []
for (i, (doc, doc_count)) in enumerate(get_docs_and_counts(outdir)):
    doc_tfidf.append({})
    for (index, count) in doc.iteritems():
        # tf = float(count) / doc_count
        tf = float(count)
        idf = index_idf[index]
        doc_tfidf[i][index] = tf * idf
        # doc_tfidf[i][index] = idf

print('Done!')

idf_out = os.path.join(outdir, 'idf')
tfidf_out = os.path.join(outdir, 'train.ind')
vocab_out = os.path.join(outdir, 'vocab')

print('Writing idf file... ', end='')
with open(idf_out, 'w') as fout:
    for (index, idf) in index_idf.iteritems():
        s_out = ' '.join([str(index), str(idf), '\n'])
        fout.write(s_out)
print('Done!')

print('Writing train.ind file... ', end='')
with open(tfidf_out, 'w') as fout:
    for (i, doc) in enumerate(doc_tfidf, 1):
        for (index, tfidf) in sorted(doc.iteritems(), key=lambda x: x[0]):
            s_out = '\t'.join([str(index+1), str(i), str(tfidf), '\n'])
            fout.write(s_out)
print('Done!')

print('Writing vocab file... ', end='')
with open(vocab_out, 'w') as fout:
    for (word, index) in sorted(word_to_index.iteritems(), key=lambda x: x[1]):
        s_out = ' '.join([word, str(index), '\n'])
        fout.write(s_out)
print('Done!')









