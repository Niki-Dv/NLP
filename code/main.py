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


if __name__ == '__main__':
    
    #L.fit( join(data_path, 'test1.wtag'))
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = join(curr_dir, "..", 'data')
    train_file_1 = join(data_path, 'train1.wtag')
    L=LLM(1000,4999,data_path)
    L.train(train_file_1)
    #Viterbi_test(L)
    #cProfile.runctx("Viterbi_test(L)",{"Viterbi_test":Viterbi_test},{"L": L},"qe")
    test_file_1= join(data_path, 'test1.wtag')
    comp_file_1= join(data_path, 'comp1.words')
    test_file_without_tags_1= join(data_path, 'test1.words')
    if not  os.path.isfile(test_file_without_tags_1):
        sperate_tags(test_file_1,test_file_without_tags_1)
    L.tag_file('test1.words')
    L.tag_file('comp1.words')






