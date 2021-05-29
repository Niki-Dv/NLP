import queue
import sys, os
from os.path import join
import re
import numpy as np
import feature_classes
import pickle
import scipy.optimize
import scipy
import time
import random
import multiprocessing
class LLM():

    def __init__(self, feat_thresh, special_features_thresh, num_line_iter, data_path, save_files_prefix="", viterbi_beam_num=5,optim_lambda_val = 0.3):
        """

        :param feat_thresh:feature count threshold - empirical count must be higher than this
        :param num_line_iter: number of line iteraions for loss func
        :param data_path: data folder path
 
        """

        self.feat_thresh = feat_thresh#
        self.optim_lambda_val = optim_lambda_val
        self.special_feat_threshold = special_features_thresh
        self.m = viterbi_beam_num
        self.num_line_iter= num_line_iter# 
        self.data_path=data_path#
        self.save_files_prefix = save_files_prefix
        self.feat_stats = None

    def loss_function_gradient(self,w, train_total_dict, tot_num_features, OPTIM_LAMBDA=0.5):
        """

        :param w:
        :param feat_class:
        :param train_file_path:
        :param OPTIM_LAMBDA:
        :return: each line dictionary is saved in the following format
        """
        start = time.time()

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

        num_lines = max(train_total_dict.keys())
        curr_lines_idxs = random.sample(list(range(num_lines)), min(self.num_line_iter, num_lines))

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
        train_data_path = join(self.data_path, f'{self.save_files_prefix}_train_data_dict_'
                                                    f'{self.feat_thresh}_lambda_{self.optim_lambda_val}_'
                                                    f'{self.feat_stats.n_total_features}.pkl')

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
        t0 = time.time()
        self.feat_stats = feature_classes.feature_statistics_class()
        self.feat_stats.get_all_counts(self.train_file_path)
        print('finished creating features statistic dicts')
        self.feat_class = feature_classes.feature2id_class(self.feat_stats, self.feat_thresh, self.special_feat_threshold)
        self.feat_class.get_all_feat_dicts(self.train_file_path)
        print(f'finished creating features dicts time took to create final dict for train file = {time.time()-t0}')
         #get feat_class

        optimal_weights_path = join(self.data_path, f'{self.save_files_prefix}_optimal_weights_'
                                                    f'{self.feat_thresh}_lambda_{self.optim_lambda_val}_'
                                                    f'{self.feat_stats.n_total_features}.pkl')

        if os.path.isfile(optimal_weights_path):
            print('Weights already calculated')
            with open(optimal_weights_path, 'rb') as f:
                return pickle.load(f)

        tot_num_features = self.feat_class.n_total_features
        total_train_data_dict = self.create_feature_representation_dict()
        print('finished creating all data representation dict')
        # find optimal weights
        w_0 = np.random.rand(tot_num_features)
        t0 = time.time()
        res = scipy.optimize.minimize(self.loss_function, w_0,
                                    args=(total_train_data_dict, tot_num_features, self.optim_lambda_val),
                                    method="L-BFGS-B",
                                    jac=self.loss_function_gradient)
        with open(optimal_weights_path, 'wb') as f:
            pickle.dump(res.x, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'saved optimal weights in: {optimal_weights_path}')
        print(f"time took for calculating optimal weights = {time.time() -t0}")
        return res.x

    ################################################################################################
    def train(self,file_path):
        self.train_file_path=file_path
        self.w = self.find_optimal_weights()

    ################################################################################################
    def Viterbi(self, line):
        """
            gets weight vector w and list of words s and and m as beam size returns
            :return: the most plausible  tag sequence for s  using viterbi
        """
        s = line.split()
        st= time.time()
        pi= {}
        Bp={}
        Bp[0]={}
        pi[0]={}
        pi[0]['*','*']=1
        end=len(s)+1
        t_k=lambda k: ['.'] if k>=end  else ( list(self.feat_class.possible_tags) if k > 0   else ['*'] )
        # getting group of tag at k level
        get_w=lambda l: [0] if len(l)==0 else self.w[l]

        f2= np.vectorize( lambda w,p_2_tag,p_tag,t,wn,wp:np.array([w,p_2_tag,p_tag,t,wn,wp],dtype=object) ,excluded=['w','wn',"wp"],signature='(),(),(),(),(),()->(k)')
        #create history vector
        f12= np.vectorize( lambda y:  np.sum(get_w(self.feat_class.get_represent_input_with_features(y))) ,signature='(6)->()')
        #sum the features weights getting from history vector
        f3= np.vectorize( (lambda k,h,x: pi[k][h[1],h[2]]*x) , excluded = ['k'] ,signature='(),(6),()->()')

        def foo(h,x,d1,d2):
            d1[h[2],h[3]]= x
            d2[h[2],h[3]]= h[1]
        f4= np.vectorize( foo , excluded=['d1','d2'], signature='(6),(),(),()->()')

        tp_2=["*"]
        tp_1=["*"]
        for k in range(1,len(s)+1):
            Bp[k]={}
            pi[k]={}
            w= s[k-1]
            wn="." if k>=end-1 else s[k]
            wp= s[k-2]
            t=t_k(k)
            x,y,z=np.meshgrid(tp_2,tp_1,t)
            b=f2(w,x,y,z,wn,wp)
            c=f12(b)
            c=scipy.special.softmax(c,axis=2)
            c=f3(k-1,b,c)
            max_indices=np.argmax(c,axis=1)
            I, J = np.indices(max_indices.shape)
            cq=c[I,max_indices,J]
            bq=b[I,max_indices,J]
            indexs=np.argpartition(cq,kth=cq.shape[-1]- self.m ,axis=-1)[..., -self.m:]
            cq=cq[...,indexs]
            bq=bq[...,indexs,:]
            tp_2=np.unique(bq[...,2])
            tp_1=np.unique(bq[...,3])
            d1={}
            d2={}
            f4(bq,cq,d1,d2)
            Bp[k]=d2
            pi[k]=d1

        tag_s= [max(p_k.items(), key=lambda x:x[1])[0] for k,p_k in pi.items()]
        n= len(s)
        s_tags=list(tag_s[-1])

        while n>2:
            s_tags.insert(0,Bp[n][s_tags[0],s_tags[1]])# workes better
            n-=1

        return s_tags


    ################################################################################################
    def tag_file_multi(self,file_name):
        """
        gets file_name in data path with sentences
         tags every word and save it in data_path\\tags_{feat_thresh}_file_name.  
        
        """
        t0 = time.time()
        head=file_name.split(".")[0]
        save_path=join(self.data_path, f'{self.save_files_prefix}_{head}_tags_'
                                                    f'{self.feat_thresh}_lambda_{self.optim_lambda_val}_'
                                                    f'{self.feat_stats.n_total_features}_vit_m_{self.m}.wtag')
        file_path= join(self.data_path,file_name)
    
        with open(file_path) as f_r:
            all_lines = f_r.readlines()

        print(f'pool is using  {round(multiprocessing.cpu_count()/2)} processes')
        with multiprocessing.Pool(processes=round(multiprocessing.cpu_count()/2)) as pool:
            results = pool.map(self.Viterbi, all_lines)

        with open(file_path) as f_r, open(save_path, "w+") as f_s:
            for line_idx, line in enumerate(f_r):
                splited_words = line.split()
                # del splited_words[-1] could be needed
                tags = results[line_idx]
                s = " ".join(list(map(lambda x, y: x + "_" + y, splited_words, tags)))
                f_s.write(s + "\n")
                f_s.flush()


        print(f"saved in {save_path}, tagging took: {time.time()-t0}")
        return save_path

    ################################################################################################
    def tag_file_multi_2(self, file_name):
        """
        gets file_name in data path with sentences
         tags every word and save it in data_path\\tags_{feat_thresh}_file_name.

        """
        t0 = time.time()
        head=file_name.split(".")[0]
        save_path=join(self.data_path, f'{self.save_files_prefix}_{head}_tags_'
                                                    f'{self.feat_thresh}_lambda_{self.optim_lambda_val}_'
                                                    f'{self.feat_stats.n_total_features}_vit_m_{self.m}.wtag')
        file_path = join(self.data_path, file_name)

        with open(file_path) as f_r:
            all_lines = f_r.readlines()

        lines_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        print(f'pool is using  {round(multiprocessing.cpu_count() / 2)} processes')
        the_pool = multiprocessing.Pool(round(multiprocessing.cpu_count() / 2), worker_main, (self, lines_queue, results_queue,))
        for line_item in enumerate(all_lines):
            lines_queue.put(line_item)

        f_s = open(save_path, "w+")
        f_s.close()

        results_dict = {}
        curr_line_needed = 0
        while True:
            if curr_line_needed in results_dict.keys():
                f_s = open(save_path, "a")
                f_s.write(results_dict[curr_line_needed] + "\n")
                f_s.close()
                curr_line_needed += 1
                if curr_line_needed == len(all_lines):
                    break
            try:
                line_idx, s = results_queue.get(timeout=50)
                results_dict[line_idx] = s
            except queue.Empty:
                pass

        print(f"saved in {save_path}, tagging took: {time.time() - t0}")
        return save_path

    ################################################################################################
    def tag_file(self, file_name):
        """
        gets file_name in data path with sentences
         tags every word and save it in data_path\\tags_{feat_thresh}_file_name.

        """
        save_path = join(self.data_path, f'tags_{self.save_files_prefix}{self.feat_thresh}_{file_name}')
        file_path = join(self.data_path, file_name)
        with open(file_path) as f_r, open(save_path, "w+") as f_s:
            for line_idx, line in enumerate(f_r):
                splited_words = line.split()
                # del splited_words[-1] could be needed
                tags = self.Viterbi(line)
                s = " ".join(list(map(lambda x, y: x + "_" + y, splited_words, tags)))
                print(f"writing line {line_idx}")
                print(s)
                f_s.write(s + "\n")
                f_s.flush()

        print("saved in {save_path}")

################################################################################################
def worker_main(L, lines_queue, results_queue):
    while True:
        line_idx, line = lines_queue.get()
        tags = L.Viterbi(line)
        splited_words = line.split()
        s = " ".join(list(map(lambda x, y: x + "_" + y, splited_words, tags)))
        results_queue.put((line_idx, s))