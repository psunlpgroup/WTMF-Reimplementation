

import os
import sys


def is_valid_file(fpath):
    return os.path.isfile(fpath) and os.path.basename(fpath)[0] != '.'


def read_chunks(indir):
    fpaths = [os.path.join(indir, f) for f in os.listdir(indir)]
    fpaths = filter(is_valid_file, fpaths)
    for (chunk_i, fpath) in enumerate(fpaths, 1):
        with open(fpath, 'r') as fin:
            for line in fin:
                yield (line, chunk_i)


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


def find_coverage(indir, ref_vocab, threshold):
    vocab = set()
    ref_vocab = set(ref_vocab)
    for (line, i) in read_chunks(indir):
        tokens = line.split()
        for token in tokens:
            if token not in vocab and token in ref_vocab:
                vocab.add(token)
                coverage = float(len(vocab)) / len(ref_vocab)
                if coverage >= threshold:
                    return (i, coverage)
    return (i, coverage)


ref_indir = sys.argv[1]
new_indir = sys.argv[2]
if len(sys.argv) > 3:
    threshold = float(sys.argv[3])
else:
    threshold = 1.0
if len(sys.argv) > 4:
    min_word_count = int(sys.argv[4])
else:
    min_word_count = 1

ref_vocab = build_vocab(ref_indir, min_word_count)
(chunks, coverage) = find_coverage(new_indir, ref_vocab, threshold)
print('{:.2f}% coverage in {} chunks'.format(coverage*100, chunks))















