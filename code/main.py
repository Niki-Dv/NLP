import sys, os
from os.path import join
import re
import numpy as np
from LLM import LLM
import pickle
import scipy.optimize
import time
import random
import cProfile
import pathlib
curr_dir = pathlib.Path(__file__).parent.absolute()
data_path = join(curr_dir, '..', 'data')

def Viterbi_test(L):
    line="The_DT plant_NN employs_VBZ between_IN 800_CD and_CC 900_CD on_IN three_CD shifts_NNS ._."
    splited_words = re.split('[ \n]', line)
    del splited_words[-1]
    words, tags = [re.split('_', word)[0] for word in splited_words], [re.split('_', word)[1] for word in splited_words] 
    tags_p=L.Viterbi(words)
    r= np.array(tags_p)== np.array(tags)
    print(r)

def sperate_tags(file_path_read,file_path_write):
    """
    get file_path_read of file with words and tags write's to file_path_write only words
    """
    with open(file_path_read) as f_r, open(file_path_write,"w+") as f_s:
            for line in f_r:
                splited_words = re.split('[ \n]', line)
                words = [re.split('_', word)[0] for word in splited_words] 
                f_s.write(" ".join(words)+"\n")

def test(data_path, your_tags_file):
    file_1=your_tags_file
    file_2=join(data_path, 'test1.wtag')
    dict_res_by_tag = {}
    with open(file_1) as f_1, open(file_2) as f_2:
        lines_1=f_1.readlines()
        lines_2=f_2.readlines()
        print(f"{len(lines_1)}, {len(lines_2)}")
        true_count = 0
        false_count = 0
        for i in range(len(lines_1)):

            splited_words2 = re.split('[ \n]', lines_2[i])
            splited_words1 = re.split('[ \n]', lines_1[i])

            del splited_words1[-1]
            del splited_words2[-1]
            for w_idx in range(len(splited_words1)-1):
                cur_word_1, cur_tag_1 = splited_words1[w_idx].split('_')
                cur_word_2, cur_tag_2 = splited_words2[w_idx].split('_')
                if cur_tag_2 not in dict_res_by_tag.keys():
                    dict_res_by_tag[cur_tag_2] = {}
                    dict_res_by_tag[cur_tag_2][True] = 0
                    dict_res_by_tag[cur_tag_2][False] = 0
                    dict_res_by_tag[cur_tag_2]['Total'] = 0
                if cur_tag_1 != cur_tag_2:
                     dict_res_by_tag[cur_tag_2][False] +=1
                     false_count +=1
                else:
                    dict_res_by_tag[cur_tag_2][True] +=1
                    true_count +=1

                dict_res_by_tag[cur_tag_2]['Total'] +=1

    dict_res_by_tag = dict(sorted(dict_res_by_tag.items(), key=lambda item: item[1]['Total']))
    for pos_tag, tag_dict in dict_res_by_tag.items():
        if tag_dict[True]/(tag_dict[True] + tag_dict[False]) * 100 < 95:
            print(f'Results for tag {pos_tag} are correct at : {tag_dict[True]/(tag_dict[True] + tag_dict[False]) * 100} of total: {tag_dict[True] + tag_dict[False]}')

    print(f'Results for general are correct at : {true_count/(true_count + false_count) * 100}')

def train_test_section_a():
    data_path = join(curr_dir, "..", 'data')
    train_file_name = 'train1.wtag'
    train_file = join(data_path, train_file_name)
    L = LLM(10, 10, 4000, data_path, train_file_name)
    L.train(train_file)
    tags_file = L.tag_file_multi_2('niki_check')
    test(data_path, tags_file)

def train_test_section_b():
    data_path = join(curr_dir, "..", 'data')
    train_file_name = 'train2.wtag'
    train_file = join(data_path, train_file_name)
    L = LLM(10, 10, 4999, data_path, train_file_name)
    L.train(train_file)
    tags_file = L.tag_file_multi_2('comp1.words')
    tags_file = L.tag_file_multi_2('comp2.words')

if __name__ == '__main__':
    #train_test_section_a()
    #train_test_section_b()

    data_path = join(curr_dir, "..", 'data')
    # train_file_1 = join(data_path, 'train1.wtag')
    # train_file_2 = join(data_path, 'train2.wtag')
    # L = train_on_data(train_file_1, 10, 'train2.wtag')
    #Viterbi_test(L)
    #cProfile.runctx("Viterbi_test(L)",{"Viterbi_test":Viterbi_test},{"L": L})
    # test_file_1= join(data_path, 'test1.wtag')
    # comp_file_1= join(data_path, 'comp1.words')
    # test_file_without_tags_1= join(data_path, 'v2test1.words')
    # if not  os.path.isfile(test_file_without_tags_1):
    #     sperate_tags(test_file_1,test_file_without_tags_1)
    data_path = join(curr_dir, "..", 'data')
    my_tags = join(curr_dir, "..",'data', 'tags_10_v2test1.words')
    test(data_path, my_tags)






