from collections import OrderedDict
import sys
import re
import numpy as np
from collections import Counter

class feature_statistics_class():
    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict =        OrderedDict()
        self.suffix_tags_count_dict =       OrderedDict()
        self.prefix_tags_count_dict =       OrderedDict()
        self.tags_chain_count_dict =        OrderedDict()
        self.tags_chain_len_2_count_dict =  OrderedDict()
        self.tag_count_dict =               OrderedDict()
        self.prev_word_tag_count_dict =     OrderedDict()
        self.next_word_tag_count_dict =     OrderedDict()

        # ---Add more count dictionaries here---

    #######################################################################################
    def get_all_counts(self, file_path):
        self.get_word_tag_pair_count(file_path)
        self.get_suffix_tag_pair_count(file_path)
        self.get_prefix_tag_pair_count(file_path)
        self.get_tags_chain_count(file_path)
        self.get_tags_chain_len_2_count(file_path)
        self.get_tag_count(file_path)
        self.get_prev_word_tag_pair_count(file_path)
        self.get_next_word_tag_pair_count(file_path)

    #######################################################################################
    def get_word_tag_pair_count(self, file_path):  # feature 100
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(cur_word, cur_tag)] = 1
                    else:
                        self.words_tags_count_dict[(cur_word, cur_tag)] += 1

    #######################################################################################
    def get_suffix_tag_pair_count(self, file_path):  # feature 101
        """
            Extract out of text all suffix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    w = ""
                    for l in cur_word[-4:][::-1]:
                        w = l + w
                        if (w, cur_tag) not in self.suffix_tags_count_dict:
                            self.suffix_tags_count_dict[(w, cur_tag)] = 1
                        else:
                            self.suffix_tags_count_dict[(w, cur_tag)] += 1

    #######################################################################################
    def get_prefix_tag_pair_count(self, file_path):  # feature 102
        """
            Extract out of text all prefix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    w = ""
                    for l in cur_word[:4]:
                        w = w + l
                        if (w, cur_tag) not in self.prefix_tags_count_dict:
                            self.prefix_tags_count_dict[(w, cur_tag)] = 1
                        else:
                            self.prefix_tags_count_dict[(w, cur_tag)] += 1

    #######################################################################################
    def get_tags_chain_count(self, file_path):  # feature 103
        """
            Extract out of text all tags chain
            :param file_path: full path of the file to read
                return all  tags chain with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                tags_list = [re.split('_', word)[1] for word in splited_words]
                history = ["*", "*"]  # starting tags
                for cur_tag in tags_list:
                    key = (history[0], history[1], cur_tag)
                    if key not in self.tags_chain_count_dict:
                        self.tags_chain_count_dict[key] = 1
                    else:
                        self.tags_chain_count_dict[key] += 1
                    history[0] = history[1]
                    history[1] = cur_tag

    #######################################################################################
    def get_tags_chain_len_2_count(self, file_path):  # feature 104
        """
            Extract out of text all prefix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                prev_tag = "*"
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (prev_tag, cur_tag) not in self.tags_chain_len_2_count_dict:
                        self.tags_chain_len_2_count_dict[(prev_tag, cur_tag)] = 1
                    else:
                        self.tags_chain_len_2_count_dict[(prev_tag, cur_tag)] += 1

                    prev_tag = cur_tag

    #######################################################################################
    def get_tag_count(self, file_path):  # feature 105
        """
            Extract out of text all tag 
            :param file_path: full path of the file to read
                return all tag with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if cur_tag not in self.tag_count_dict:
                        self.tag_count_dict[cur_tag] = 1
                    else:
                        self.tag_count_dict[cur_tag] += 1

    #######################################################################################
    def get_prev_word_tag_pair_count(self, file_path):  # feature 106
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                prev_word = '*'
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (prev_word, cur_tag) not in self.prev_word_tag_count_dict:
                        self.prev_word_tag_count_dict[(prev_word, cur_tag)] = 1
                    else:
                        self.prev_word_tag_count_dict[(prev_word, cur_tag)] += 1
                    prev_word = cur_word

    #######################################################################################
    def get_next_word_tag_pair_count(self, file_path):  # feature 107
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                words, tags = [re.split('_', word)[0] for word in splited_words], [re.split('_', word)[1] for word in splited_words]
                next_word_and_tag = list(zip(words[1:], tags[:-1]))
                for w, t in next_word_and_tag:
                    if (w, t) not in self.next_word_tag_count_dict:
                        self.next_word_tag_count_dict[(w, t)] = 1
                    else:
                        self.next_word_tag_count_dict[(w, t)] += 1


#######################################################################################################################
class feature2id_class():
    def __init__(self, feature_statistics, threshold, special_feat_threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this
        self.special_feat_threshold = special_feat_threshold  # feature count threshold - empirical count must be higher than this
        self.special_tags = ['CD', 'JJR', 'MD', '""', "''", "``"]
        self.possible_tags = set()
        self.n_total_features = 0  # Total number of features accumulated

        self.n_word_tag_pairs = 0  # Number of Word\Tag pairs features
        self.n_suffix_tag_pairs = 0  # Number of suffix\Tag pairs features
        self.n_prefix_tag_pairs = 0  # Number of prefix\Tag pairs features
        self.n_chain_tags = 0  # Number of chain of tags features
        self.n_chain_tags_len_2 = 0  # Number of chain of tags features
        self.n_tag = 0
        self.n_prev_word_tag_pairs = 0
        self.n_next_word_tag_pairs = 0
        self.n_is_nunber = 1
        self.n_is_capital_in_middle = 1
        self.n_is_word_len_smaller_than_4 = 1
        self.n_is_word_smaller_than_prev = 1
        self.n_is_word_smaller_than_next = 1
        self.n_is_word_ends_with_s = 1
        self.n_word_is_not_word_or_number = 1
        self.n_bias_feature = 1

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.suffix_tags_dict = OrderedDict()
        self.prefix_tags_dict = OrderedDict()
        self.tags_chain_dict = OrderedDict()
        self.tags_chain_len_2_dict = OrderedDict()
        self.tag_dict = OrderedDict()
        self.prev_word_tag_dict = OrderedDict()
        self.next_word_tag_pairs_dict = OrderedDict()

    #######################################################################################
    def get_all_feat_dicts(self, file_path):
        self.get_word_tag_pairs(file_path)
        self.get_suffix_tag_pairs(file_path)
        self.get_prefix_tag_pairs(file_path)
        self.get_chain_tags(file_path)
        self.get_tags_chain_len_2(file_path)
        self.get_tag(file_path)
        self.get_prev_word_tag_pair(file_path)
        self.get_next_word_tag_pairs(file_path)
        self.n_total_features += 1 # for number feat
        self.n_total_features += 1 # for is captial feat
        self.n_total_features += 1 # for is word length smaller then 4
        self.n_total_features += 1 # for is word smaller then prev word
        self.n_total_features += 1 # for is word smaller then next word
        self.n_total_features += 1 # for word ends with s
        self.n_total_features += 1 # for word is not actual word or number
        self.n_total_features += 1 # for word has all capital letters
        self.n_total_features += 1 # for bias

        num_feat_list = [
            self.n_word_tag_pairs,
            self.n_suffix_tag_pairs,
            self.n_prefix_tag_pairs,
            self.n_chain_tags,
            self.n_chain_tags_len_2,
            self.n_tag,
            self.n_prev_word_tag_pairs,
            self.n_next_word_tag_pairs,
            self.n_is_nunber,
            self.n_is_capital_in_middle,
            self.n_is_word_len_smaller_than_4,
            self.n_is_word_smaller_than_prev,
            self.n_is_word_smaller_than_next,
            self.n_is_word_ends_with_s,
            self.n_word_is_not_word_or_number,
            self.n_bias_feature
        ]
        for n_feat in num_feat_list:
            print(n_feat)

    def get_pos_of_captial_in_mid_feat(self):
        num_feat_list = [
            self.n_word_tag_pairs,
            self.n_suffix_tag_pairs,
            self.n_prefix_tag_pairs,
            self.n_chain_tags,
            self.n_chain_tags_len_2,
            self.n_tag,
            self.n_prev_word_tag_pairs,
            self.n_next_word_tag_pairs,
            self.n_is_nunber
        ]

        return np.sum(num_feat_list)

    def get_pos_of_is_number(self):
        num_feat_list = [
            self.n_word_tag_pairs,
            self.n_suffix_tag_pairs,
            self.n_prefix_tag_pairs,
            self.n_chain_tags,
            self.n_chain_tags_len_2,
            self.n_tag,
            self.n_prev_word_tag_pairs,
            self.n_next_word_tag_pairs,
        ]
        return np.sum(num_feat_list)

    #######################################################################################
    def get_represent_input_with_features(self, history):
        """
            Extract feature vector in per a given history
            :param history: touple{word, previous tag (-2), previous tag (-1), ctag, nword, pword}
            :param word_tags_dict: word\tag dict
                Return a list with all features that are relevant to the given history
        """
        num_feat_list = [
            self.n_word_tag_pairs,
            self.n_suffix_tag_pairs,
            self.n_prefix_tag_pairs,
            self.n_chain_tags,
            self.n_chain_tags_len_2,
            self.n_tag,
            self.n_prev_word_tag_pairs,
            self.n_next_word_tag_pairs,
            self.n_is_nunber,
            self.n_is_capital_in_middle,
            self.n_is_word_len_smaller_than_4,
            self.n_is_word_smaller_than_prev,
            self.n_is_word_smaller_than_next,
            self.n_is_word_ends_with_s,
            self.n_word_is_not_word_or_number,
            self.n_bias_feature
        ]

        word = history[0]
        p_2_tag = history[1]
        p_tag = history[2]
        ctag = history[3]
        nword = history[4]
        pword = history[5]

        features = []

        # feature 100
        feat_dict_count = 0
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        if (word, ctag) in self.words_tags_dict:
            features.append(feat_dict_start_point + self.words_tags_dict[(word, ctag)])

        # feature 101
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        w = ""
        for l in word[-4:][::-1]:
            w = l + w
            if (word, ctag) in self.suffix_tags_dict:
                features.append(feat_dict_start_point + self.suffix_tags_dict[(w, ctag)])

        # feature 102
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        w = ""
        for l in word[:4]:
            w = w + l
            key = (w, ctag)
            if key in self.prefix_tags_dict:
                features.append(feat_dict_start_point + self.prefix_tags_dict[key])

        # feature 103
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        key = (p_2_tag, p_tag, ctag)
        if key in self.tags_chain_dict:
            features.append(feat_dict_start_point + self.tags_chain_dict[key])

        # feature 104
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        key = (p_tag, ctag)
        if key in self.tags_chain_len_2_dict:
            features.append(feat_dict_start_point + self.tags_chain_len_2_dict[key])

        # feature 105
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        key = (ctag)
        if key in self.tag_dict:
            features.append(feat_dict_start_point + self.tag_dict[key])

        # feature 106
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        key = (pword, ctag)
        if key in self.prev_word_tag_dict:
            features.append(feat_dict_start_point + self.prev_word_tag_dict[key])

        # feature 107
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        key = (nword, ctag)
        if key in self.next_word_tag_pairs_dict:
            features.append(feat_dict_start_point + self.next_word_tag_pairs_dict[key])

        # feature numbers
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        if self.get_is_number(word) and ctag == 'CD':
            features.append(feat_dict_start_point)

        # feature is cpaital in middle
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        if self.get_is_capital_in_middle(word) and pword != "*" and (ctag == "NNP" or ctag == "NNPS"):
            features.append(feat_dict_start_point)

        # feature for is word length smaller than 4
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        TAGS_FOR_SHORT_WORDS = ['PRP', "CD", "IN"]
        if len(word) < 4 and ctag in TAGS_FOR_SHORT_WORDS:
            features.append(feat_dict_start_point)

        # feature for is word length smaller than prev
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        TAGS_FOR_SHORT_WORDS = ['PRP', "CD", "IN"]
        if len(word) < len(pword) and ctag in TAGS_FOR_SHORT_WORDS:
            features.append(feat_dict_start_point)

        # feature for is word length smaller than next
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        TAGS_FOR_SHORT_WORDS = ['PRP', "CD", "IN"]
        if len(word) < len(nword) and ctag in TAGS_FOR_SHORT_WORDS:
            features.append(feat_dict_start_point)

        # feature for is word ends with s
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        TAGS_FOR_ENDS_WITH_S = ['NNS', "NNPS"]
        if word[-1] == 's' and ctag in TAGS_FOR_ENDS_WITH_S:
            features.append(feat_dict_start_point)

        # feature for is word not word or number
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        if self.get_is_not_word_or_number(word) and word == ctag:
            features.append(feat_dict_start_point)

        # feature bias
        feat_dict_count += 1
        feat_dict_start_point = int(np.sum(num_feat_list[:feat_dict_count]))
        features.append(feat_dict_start_point)

        assert len(Counter(features).keys()) == len(features), 'same feature number multiple times'
        return features

    #######################################################################################
    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) and\
                            ((self.feature_statistics.words_tags_count_dict[(cur_word, cur_tag)] >= self.threshold) or
                             (cur_tag in self.special_tags)):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_word_tag_pairs
                        self.n_word_tag_pairs += 1
        self.n_total_features += self.n_word_tag_pairs

    #######################################################################################
    def get_suffix_tag_pairs(self, file_path):
        """
            Extract out of text all suffix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    w = ""
                    for l in cur_word[-4:][::-1]:
                        w = l + w
                        if ((w, cur_tag) not in self.suffix_tags_dict) \
                                and (self.feature_statistics.suffix_tags_count_dict[(w, cur_tag)] >= self.threshold):
                            self.words_tags_dict[(w, cur_tag)] = self.n_suffix_tag_pairs
                            self.n_suffix_tag_pairs += 1

        self.n_total_features += self.n_suffix_tag_pairs

    #######################################################################################
    def get_prefix_tag_pairs(self, file_path):  # feature 102
        """
            Extract out of text all prefix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    w = cur_word[0]
                    for l in cur_word[1:3]:
                        w = w + l
                        key = (w, cur_tag)
                        if key not in self.prefix_tags_dict and \
                                self.feature_statistics.prefix_tags_count_dict[key] > self.threshold:
                            self.prefix_tags_dict[key] = self.n_prefix_tag_pairs
                            self.n_prefix_tag_pairs += 1

            self.n_total_features += self.n_prefix_tag_pairs

    #######################################################################################
    def get_chain_tags(self, file_path):
        """
            Extract out of text all chain tags 
            :param file_path: full path of the file to read
                return all chain tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                tags_list = [re.split('_', word)[1] for word in splited_words]
                history = ["*", "*"]  # starting tags
                for cur_tag in tags_list:
                    key = (history[0], history[1], cur_tag)
                    if key not in self.tags_chain_dict and \
                            self.feature_statistics.tags_chain_count_dict[key] >= self.threshold:
                        self.tags_chain_dict[key] = self.n_chain_tags
                        self.n_chain_tags += 1
                    history[0] = history[1]
                    history[1] = cur_tag

        self.n_total_features += self.n_chain_tags

    #######################################################################################
    def get_tags_chain_len_2(self, file_path):  # feature 104
        """
            Extract out of text all prefix/tag pairs
            :param file_path: full path of the file to read
                return all suffix/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                prev_tag = "*"
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    key = (prev_tag, cur_tag)
                    if key not in self.tags_chain_len_2_dict and \
                        self.feature_statistics.tags_chain_len_2_count_dict[key] >= self.threshold:
                        self.tags_chain_len_2_dict[key] = self.n_chain_tags_len_2
                        self.n_chain_tags_len_2 += 1
                    prev_tag = cur_tag

            self.n_total_features += self.n_chain_tags_len_2

    #######################################################################################
    def get_tag(self, file_path):
        """
            Extract out of text all tags 
            :param file_path: full path of the file to read
                return all tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]

                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    self.possible_tags.add(cur_tag)
                    if (cur_tag not in self.tag_dict) \
                            and (self.feature_statistics.tag_count_dict[cur_tag] >= self.threshold):
                        self.tag_dict[cur_tag] = self.n_tag
                        self.n_tag += 1
        self.n_total_features += self.n_tag

    #######################################################################################
    def get_prev_word_tag_pair(self, file_path):  # feature 106
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                prev_word = '*'
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    key = (prev_word, cur_tag)
                    if key not in self.prev_word_tag_dict and \
                            self.feature_statistics.prev_word_tag_count_dict[key] >= self.threshold:
                        self.prev_word_tag_dict[key] = self.n_prev_word_tag_pairs
                        self.n_prev_word_tag_pairs += 1
                    prev_word = cur_word

            self.n_total_features += self.n_prev_word_tag_pairs

    #######################################################################################
    def get_next_word_tag_pairs(self, file_path): # feature 107
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]
                words, tags = [re.split('_', word)[0] for word in splited_words], [re.split('_', word)[1] for word in
                                                                                   splited_words]
                next_word_and_tag = list(zip(words[1:], tags[:-1]))
                for w, t in next_word_and_tag:
                    if ((w, t) not in self.next_word_tag_pairs_dict) \
                            and (self.feature_statistics.next_word_tag_count_dict[(w, t)] >= self.threshold):
                        self.words_tags_dict[(w, t)] = self.n_next_word_tag_pairs
                        self.n_next_word_tag_pairs += 1
        self.n_total_features += self.n_next_word_tag_pairs

    ######################################################################################
    def get_is_number(self, word):
        reg = re.compile("-?[0-9][0-9,.]+")
        return bool(reg.match(word))

    ######################################################################################
    def get_is_capital_in_middle(self, word):
        return word[0].isupper() and not np.all(list(map(str.isupper, word)))

    ######################################################################################
    def get_is_not_word_or_number(self, word):
        if not self.get_is_number(word) and not bool(re.match('[a-z,A-Z]', word)):
            return True
        return False

    ######################################################################################
    def get_all_letters_are_capital(self, word):
        return np.all(list(map(str.isupper, word)))