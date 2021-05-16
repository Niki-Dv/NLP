import sys, os
from os.path import join
import re
import numpy as np
import feature_classes
import pickle
import scipy.optimize
import time
import random

random.seed(100)
FEAT_THRESH = 1000
NUM_LINES_ITER = 4999

######################################################################################################
def loss_function_gradient(w, train_total_dict, tot_num_features, OPTIM_LAMBDA=0.5):
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
    curr_lines_idxs = random.sample(list(range(num_lines)), NUM_LINES_ITER)

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
def loss_function(w, train_total_dict, tot_num_features, OPTIM_LAMBDA = 0.5):
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
    curr_lines_idxs = random.sample(list(range(num_lines)), NUM_LINES_ITER)

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
def create_feature_representation_dict(file_path, feat_class):
    """
    create dictionary s.t.: dict[<line index>][<word idx in current line>] = [<actual feature>,
    <feature for some tag 1>, , <feature for some tag 12> ... <feature for some tag n> ]
    :param file_path:
    :param feat_class:
    :return:
    """
    train_data_path = join(r"C:\git-projects\NLP\data", f"train_data_dict_{FEAT_THRESH}.pkl")
    if os.path.isfile(train_data_path):
        with open(train_data_path, 'rb') as f:
            total_train_data_dict = pickle.load(f)

        return total_train_data_dict

    total_train_data_dict = {}
    with open(file_path) as f:
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
                for tag in feat_class.possible_tags:
                    curr_history = (cur_word, p_2_tag, p_tag, tag, n_word, p_word)
                    all_pos_historys.append(feat_class.get_represent_input_with_features(curr_history))

                actual_feature_vec = feat_class.get_represent_input_with_features(act_curr_history)
                total_train_data_dict[line_idx][w_idx] = [actual_feature_vec] + all_pos_historys

                p_2_tag = p_tag
                p_tag = cur_tag
                p_word = cur_word

    with open(train_data_path, 'wb') as f:
        pickle.dump(total_train_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return total_train_data_dict

######################################################################################################
def find_optimal_weights(train_file_path, data_path):
    optimal_weights_path = join(data_path, f'optimal_weights_{FEAT_THRESH}.pkl')
    if os.path.isfile(optimal_weights_path):
        print('Weights already calculated')
        with open(optimal_weights_path, 'rb') as f:
            return pickle.load(f)

    feat_stats = feature_classes.feature_statistics_class()
    feat_stats.get_all_counts(train_file_path)
    feat_class = feature_classes.feature2id_class(feat_stats, FEAT_THRESH)
    feat_class.get_all_feat_dicts(train_file_path)
    tot_num_features = feat_class.n_total_features
    total_train_data_dict = create_feature_representation_dict(train_file_1, feat_class)

    # find optimal weights
    w_0 = np.random.rand(tot_num_features)
    res = scipy.optimize.minimize(loss_function, w_0,
                                  args=(total_train_data_dict, tot_num_features),
                                  method="L-BFGS-B",
                                  jac=loss_function_gradient)
    optimal_weights_path = join(data_path, f'optimal_weights_{FEAT_THRESH}.pkl')
    with open(optimal_weights_path, 'wb') as f:
        pickle.dump(res.x, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'saved optimal weights in: {optimal_weights_path}')

    return total_train_data_dict

######################################################################################################
if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = join(curr_dir, "..", 'data')
    train_file_1 = join(curr_dir, "..", 'data', 'train1.wtag')
    find_optimal_weights(train_file_1, data_path)









