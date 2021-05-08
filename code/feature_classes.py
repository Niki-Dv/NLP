from collections import OrderedDict
import re

class feature_statistics_class():

    def __init__(self):
        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()
        self.suffix_tags_count_dict= OrderedDict()
        self.tags_chain_count_dict=OrderedDict()
        self.tag_count_dict=OrderedDict()
        self.next_word_tag_count_dict=OrderedDict()
        # ---Add more count dictionaries here---


    def get_word_tag_pair_count(self, file_path):
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

    def get_suffix_tag_pair_count(self, file_path):
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
                    w=""
                    for l in cur_word[-4:][::-1]:
                        w=l+w
                        if (w, cur_tag) not in self.suffix_tags_count_dict:
                            self.suffix_tags_count_dict[(w, cur_tag)] = 1
                        else:
                            self.suffix_tags_count_dict[(cur_word, cur_tag)] += 1

    def get_tags_chain_count(self, file_path):
        """
            Extract out of text all tags chain
            :param file_path: full path of the file to read
                return all  tags chain with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('_[A-z]\w+', line)
                history=["*","*"]#starting tags 
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx][1:]
                    if (history[0],history[1], cur_tag) not in self.tags_chain_count_dict:
                            self.stags_chain_count_dict[(history[0],history[1], cur_tag)] = 1
                    else:
                            self.stags_chain_count_dict[(history[0],history[1], cur_tag)] += 1
                    history.pop()
                    history.append(cur_tag)

    def get_tag_count(self, file_path):
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
                    if  cur_tag not in self.words_tags_count_dict:
                        self.words_tags_count_dict[ cur_tag] = 1
                    else:
                        self.words_tags_count_dict[cur_tag] += 1

    def get_next_word_tag_pair_count(self, file_path):
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                splited_words[-1]= "STOP_"
                words,tags=list(zip(*list(map(lambda x: set(x.split('_')),splited_words))))#creating words and tags lists
                next_word_and_tag=list(zip(words[1:],tags[:-1]))

                for w,t in next_word_and_tag:
                    if (w, t) not in self.next_word_tag_count_dict:
                        self.next_word_tag_count_dict[(w, t)] = 1
                    else:
                        self.next_word_tag_count_dict[(w, t)] += 1    
    
    

    # --- ADD YOURE CODE BELOW --- #

class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold                    # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0                     # Total number of features accumulated
        self.n_tag_pairs = 0                          # Number of Word\Tag pairs features
        self.n_suffix_tag_pairs = 0                   # Number of suffix\Tag pairs features
        self.n_chain_tags = 0                         # Number of chain of tags features
        self.n_tag=0
        self.n_next_word_tag_pairs=0
        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.suffix_tags_dict = OrderedDict()
        self.tags_chain_count_dict =OrderedDict()
        self.tag_dict= OrderedDict()
        self.next_word_tag_pairs_dict =OrderedDict()
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
                    cur_word, cur_tag = splited_words[word_idx].split( '_')
                    w=""
                    for l in cur_word[-4:][::-1]:
                        w=l+w
                        if ((w, cur_tag) not in self.suffix_tags_dict) \
                                and (self.feature_statistics.suffix_tags_count_dict[(w, cur_tag)] >= self.threshold):
                            self.words_tags_dict[(w, cur_tag)] = self.n_suffix_tag_pairs
                            self.n_suffix_tag_pairs += 1
        self.n_total_features += n_suffix_tag_pairs

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
                    cur_word, cur_tag = splited_words[word_idx].split( '_')
                    if ((cur_word, cur_tag) not in self.words_tags_dict) \
                            and (self.feature_statistics.words_tags_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.words_tags_dict[(cur_word, cur_tag)] = self.n_tag_pairs
                        self.n_tag_pairs += 1
        self.n_total_features += self.n_tag_pairs

    def get_chain_tags(self, file_path):
        """
            Extract out of text all chain tags 
            :param file_path: full path of the file to read
                return all chain tags with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('_[A-z]\w+', line)
                history=["*","*"]#starting tags 
                for word_idx in range(len(splited_words)):
                    cur_tag = splited_words[word_idx][1:]
                    cur_word, cur_tag = splited_words[word_idx].split( '_')
                    if ((cur_word, cur_tag) not in self.tags_chain_count_dict) \
                            and (self.feature_statistics.tags_chain_count_dict[(cur_word, cur_tag)] >= self.threshold):
                        self.tags_chain_count_dict[(cur_word, cur_tag)] = self.n_chain_tags
                        self.n_chain_tags += 1
                    history.pop()
                    history.append(cur_tag)
        self.n_total_features += self.n_chain_tags

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
                    cur_word, cur_tag = splited_words[word_idx].split( '_')
                    if ( cur_tag not in self.tag_dict) \
                            and (self.feature_statistics.tag_count_dict[cur_tag] >= self.threshold):
                        self.tag_dict[ cur_tag] = self.n_tag
                        self.n_tag += 1
        self.n_total_features += self.n_tag

    def get_word_tag_pairs(self, file_path):
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('[ \n]', line)
                splited_words[-1]= "STOP_"
                words,tags=list(zip(*list(map(lambda x: set(x.split('_')),splited_words))))#creating words and tags lists
                next_word_and_tag=list(zip(words[1:],tags[:-1]))

                for w,t in next_word_and_tag:
                    if ((w, t) not in self.next_word_tag_pairs_dict) \
                            and (self.feature_statistics.words_tags_dict[(w, t)] >= self.threshold):
                        self.words_tags_dict[(w, t)] = self.n_next_word_tag_pairs
                        self.n_next_word_tag_pairs += 1
        self.n_total_features += self.n_next_word_tag_pairs
    