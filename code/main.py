import sys, os
from os.path import join
import re
import numpy as np
from LLM import LLM
import pickle
import scipy.optimize
import time
import random



def Viterbi_test(L):
    line="The_DT plant_NN employs_VBZ between_IN 800_CD and_CC 900_CD on_IN three_CD shifts_NNS ._."
    splited_words = re.split('[ \n]', line)
    del splited_words[-1]
    words, tags = [re.split('_', word)[0] for word in splited_words], [re.split('_', word)[1] for word in splited_words] 
    tags_p=L.Viterbi(words)
    r= np.array(tags_p)== np.array(tags)
    print(r)

if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = join(curr_dir, "..", 'data')
    train_file_1 = join(data_path, 'train1.wtag')
    L=LLM(1500,100,data_path)
    L.train(train_file_1)
    #L.fit( join(data_path, 'test1.wtag'))
    Viterbi_test(L)








