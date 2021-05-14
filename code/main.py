import sys, os
from os.path import join

import feature_classes

data_path = r"C:\git-projects\NLP\data"

if __name__ == '__main__':
    feat_class = feature_classes.feature_statistics_class()
    feat_class.get_all_counts(join(data_path, 'train1.wtag'))
    a = 3









