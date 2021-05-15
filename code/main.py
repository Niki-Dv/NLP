import sys, os
from os.path import join
import re

import feature_classes

data_path = r"C:\git-projects\NLP\data"

FEAT_THRESH = 100


def get_all_words_feat_rep(file_path):
    rep_vectors
    with open(file_path) as f:
        for line in f:
            splited_words = re.split('[ \n]', line)
            del splited_words[-1]

            prev_tag = '*'
            prev_word = '*'
            for word_idx in range(len(splited_words[:-1])):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                next_word, next_tag = splited_words[word_idx+1].split('_')


if __name__ == '__main__':
    feat_stats = feature_classes.feature_statistics_class()
    feat_stats.get_all_counts(join(data_path, 'train1.wtag'))
    feat_class = feature_classes.feature2id_class(feat_stats, FEAT_THRESH)
    feat_class.get_all_feat_dicts(join(data_path, 'train1.wtag'))

    a = 3









