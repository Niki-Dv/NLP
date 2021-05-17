import sys, os
from os.path import join
import re
import numpy as np
import feature_classes
import pickle
import scipy.optimize
import time
import random

class LLM():
    def __init__(self,feat_thresh,num_line_iter,data_path):
        """

        :param feat_thresh:feature count threshold - empirical count must be higher than this
        :param num_line_iter: number of line iteraions for loss func
        :param data_path: data folder path
 
        """

        self.feat_thresh = feat_thresh#  
        self.num_line_iter= num_line_iter# 
        self.data_path=data_path# 

    def loss_function_gradient(self,w, train_total_dict, tot_num_features, OPTIM_LAMBDA=0.5):
        """

        :param w:
        :param feat_class:
        :param train_file_path:
        :param OPTIM_LAMBDA:
        :return: each line dictionary is saved in the following format
        """
        start = time.time()
        print('gradient started')

        num_lines = max(train_total_dict.keys())
        curr_lines_idxs = random.sample(list(range(num_lines)), self.num_line_iter)

        gradient = np.zeros(tot_num_features)
        for line_idx in curr_lines_idxs:
            line_dict = train_total_dict[line_idx]
            for _, word_list in line_dict.items():
                actual_feature = word_list[0]
                all_possible_tags_features = word_list[1:]

                curr_feat_vec = np.zeros(tot_num_features)
                curr_feat_vec[actual_feature] = 1
                gradient += curr_feat_vec

                exp_vec_func = np.vectorize(
                    lambda feat_list: 1 if len(feat_list) == 0 else np.exp(np.sum((w[feat_list]))))
                exp_vals = exp_vec_func(all_possible_tags_features)
                exp_sum = np.sum(exp_vals)
                prob_vals = exp_vals/exp_sum

                for tag_idx, feat_list in enumerate(all_possible_tags_features):
                    if len(feat_list) == 0:
                        continue
                    gradient[feat_list] -= prob_vals[tag_idx]

        gradient -= OPTIM_LAMBDA * w
        gradient = -gradient
        print(f'gradient finished in {time.time()-start}')
        return gradient

######################################################################################################
    def loss_function(self,w, train_total_dict, tot_num_features, OPTIM_LAMBDA = 0.5):
        """

        :param w:
        :param feat_class:
        :param train_file_path:
        :param OPTIM_LAMBDA:
        :return: each line dictionary is saved in the following format
        """
        start = time.time()
        print('started calc loss func')

        num_lines = max(train_total_dict.keys())
        curr_lines_idxs = random.sample(list(range(num_lines)), self.num_line_iter)

        lose_func_val = 0
        for line_idx in curr_lines_idxs:
            line_dict = train_total_dict[line_idx]
            for _, word_list in line_dict.items():
                actual_feature = word_list[0]
                all_possible_tags_features = word_list[1:]
                if len(actual_feature) == 0:
                    continue
                lose_func_val += np.sum(w[actual_feature])

                vec_func = np.vectorize(
                    lambda feat_list: 1 if len(feat_list) == 0 else np.exp(np.sum((w[feat_list]))))

                exp_sum = np.sum(vec_func(all_possible_tags_features))
                lose_func_val -= np.log(exp_sum)

        lose_func_val -= 0.5 * OPTIM_LAMBDA * np.linalg.norm(w)
        lose_func_val = -lose_func_val
        print(f'finished calc loss func in: {time.time() - start}, loss value = {lose_func_val}')
        return lose_func_val

    ######################################################################################################
    def create_feature_representation_dict(self):
        """
        create dictionary s.t.: dict[<line index>][<word idx in current line>] = [<actual feature>,
        <feature for some tag 1>, , <feature for some tag 2> ... <feature for some tag n> ]
        :param file_path:
        :param feat_class:
        :return:
        """
        train_data_path = join(self.data_path, f"train_data_dict_{self.feat_thresh}.pkl")
        if os.path.isfile(train_data_path):
            with open(train_data_path, 'rb') as f:
                total_train_data_dict = pickle.load(f)
            return total_train_data_dict

        total_train_data_dict = {}
        with open(self.train_file_path) as f:
            for line_idx, line in enumerate(f):
                total_train_data_dict[line_idx] = {}
                splited_words = re.split('[ \n]', line)
                del splited_words[-1]

                p_2_tag = '*'
                p_tag = '*'
                p_word = '*'
                for w_idx in range(len(splited_words[:-1])):
                    cur_word, cur_tag = splited_words[w_idx].split('_')
                    n_word, _ = splited_words[w_idx + 1].split('_')
                    act_curr_history = (cur_word, p_2_tag, p_tag, cur_tag, n_word, p_word)

                    all_pos_historys = []
                    for tag in self.feat_class.possible_tags:
                        curr_history = (cur_word, p_2_tag, p_tag, tag, n_word, p_word)
                        all_pos_historys.append(self.feat_class.get_represent_input_with_features(curr_history))

                    actual_feature_vec = self.feat_class.get_represent_input_with_features(act_curr_history)
                    total_train_data_dict[line_idx][w_idx] = [actual_feature_vec] + all_pos_historys

                    p_2_tag = p_tag
                    p_tag = cur_tag
                    p_word = cur_word

        with open(train_data_path, 'wb') as f:
            pickle.dump(total_train_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        return total_train_data_dict


    ######################################################################################################
    def find_optimal_weights(self):
        optimal_weights_path = join(self.data_path, f'optimal_weights_{self.feat_thresh}.pkl')
        

        self.feat_stats = feature_classes.feature_statistics_class()
        self.feat_stats.get_all_counts(self.train_file_path)
        print('finished creating features statistic dicts')
        self.feat_class = feature_classes.feature2id_class(self.feat_stats, self.feat_thresh)
        self.feat_class.get_all_feat_dicts(self.train_file_path)
        print('finished creating features dicts')
         #get feat_class

        if os.path.isfile(optimal_weights_path):
            print('Weights already calculated')
            with open(optimal_weights_path, 'rb') as f:
                return pickle.load(f)

        tot_num_features = self.feat_class.n_total_features
        total_train_data_dict = self.create_feature_representation_dict()
        print('finished creating all data representation dict')
        # find optimal weights
        w_0 = np.random.rand(tot_num_features)
        res = scipy.optimize.minimize(self.loss_function, w_0,
                                    args=(total_train_data_dict, tot_num_features),
                                    method="L-BFGS-B",
                                    jac=self.loss_function_gradient)
        optimal_weights_path = join(self.data_path, f'optimal_weights_{self.feat_thresh}.pkl')
        with open(optimal_weights_path, 'wb') as f:
            pickle.dump(res.x, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saved optimal weights in: {optimal_weights_path}')
        
        return res.x

    def train(self,file_path):
        self.train_file_path=file_path
        self.w = self.find_optimal_weights()

    def Viterbi(self,s):
        """
        gets weight vector w and list of words s and returns 
        :return: the most plausible  tag sequence for s  using viterbi
        """
        st= time.time()
        all_possible_tags_features=[]
        feature_exp=lambda feat_list: 1 if  (type(feat_list) is list)  and len(feat_list) == 0 else np.exp(np.sum((self.w[feat_list])))
        vec_func =np.vectorize(feature_exp)
        t_k=lambda k: list(self.feat_class.possible_tags) if k>0 else ['*']
        pi= {}
        Bp={}
        pi[0]={}
        pi[0]['*','*']=1
        for k in range(1,len(s)+1):
            pi[k]={}
            for p_tag in t_k(k-1):
                f= lambda t,p_2_tag: self.feat_class.get_represent_input_with_features((s[k-1], p_2_tag, p_tag, t, ("." if  k==len(s) else s[k]), s[k-2]))
                #creating function for getting features vectors for p_tag   
                all_possible_tags_features= lambda p_2: [f(c_t,p_2) for c_t in t_k(k)]

                pre_calc={ p:pi[k-1][p,p_tag]/np.sum(vec_func(all_possible_tags_features(p))) for p in t_k(k-2) } 
                for t in t_k(k):
                    Pro=[pre_calc[p_2_tag]*feature_exp(f(t,p_2_tag)) for p_2_tag in t_k(k-2)]#note that sentence probability is exp to zero with k should consider adding ln
                    i=np.argmax(Pro)
                    pi[k][p_tag,t]=Pro[i]
                    Bp[k,p_tag,t]=t_k(k-2)[i]
        tag_s= [max(p_k.items(), key=lambda x:x[1])[0] for k,p_k in pi.items()]
        s_tags=list(tag_s[-1])
        n= len(s)
        while n>2:
            s_tags.insert(0,Bp[n,s_tags[0],s_tags[1]])
            n-=1
        print ("finished with viterbi in: " ,time.time()-st)
        return s_tags
    def tag_file(self,file_name):
        """
        gets file_name in data path with sentences
         tags every word and save it in data_path\\tags_{feat_thresh}_file_name.  
        
        """
        save_path=join(self.data_path,f'tags_{self.feat_thresh}_{file_name}')
        file_path= join(self.data_path,file_name)
        with open(file_path) as f_r, open(save_path,"w+") as f_s:
            for line in f_r:
                splited_words = line.split()
                #del splited_words[-1] could be needed
                tags= self.Viterbi(splited_words)
                s=" ".join(list(map(lambda x,y: x+"_"+y,splited_words,tags)))
                print("writing:")
                print(s)
                f_s.write(s+"\n")
        print("saved in {save_path}")

    