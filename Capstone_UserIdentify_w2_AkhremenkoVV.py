#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import os
import glob
import pickle
from collections import Counter
import itertools

import re
from tqdm import tqdm

import statsmodels.api as sm
import scipy.stats as sts
from statsmodels.stats.proportion import proportion_confint


# In[3]:


get_ipython().run_line_magic('pylab', 'inline')


# In[4]:


DATA = './data'
MY_DATA = './my_data'
ANSW = './answers'


# In[ ]:





# # подготовка нескольких обучающих выборок для сравнения

# In[5]:


def prepare_sparse_train_set_window(path_to_csv, path_to_site_freq, sess_len, wind_size):
    
    rows = 0
    site_cnt = []
    ret_userid_list = []
    # form dataframe names ['site1, site2......site{session_length}']
    col_names = ['site' + str(n+1) for n in range(sess_len)]

    #pkl_name = 'site_freq_' + str(re.findall('\d+', path_to_data)[0]) + 'users.pkl'
    with open(path_to_site_freq, 'rb') as f:
        freq_site_dict = pickle.load(f)


    # getting size of DataFrame
    # collect all sites ans them frequencies
    for file_csv in (sorted(glob.glob(path_to_csv + '/*.csv'))):
        temp_dataframe = pd.read_csv(file_csv, usecols=['site'])
        site_cnt += list(temp_dataframe.site.values)
        rows += round(temp_dataframe.shape[0] / wind_size + 0.499)

    site_cnt = Counter(site_cnt)

    # ceate DataFrame of known size
    ret_data = pd.DataFrame(index = range(rows), columns = col_names)
    index = 0


    for file_csv in (sorted(glob.glob(path_to_csv + '/*.csv'))):
            temp_dataframe = pd.read_csv(file_csv, usecols=['site'])
            userid = int(re.findall('\d+', file_csv)[1])


            sess_numb = round(temp_dataframe.shape[0] / wind_size + 0.499)
            for idx in range(sess_numb - 1):
                #new dict for current session
                new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}

                sess_start = idx*wind_size
                sess_end   = idx*wind_size + sess_len

                for n, site in enumerate(temp_dataframe.site.values[sess_start : sess_end]):
                    new_sess['site' + str(n+1)] = freq_site_dict[site][0]
                #new_sess['user_id'] = userid
                ret_userid_list.append(userid)
                
                #put data in prepared DataFrame
                ret_data.iloc[index] = new_sess
                index += 1


            new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}
            for n, site in enumerate(temp_dataframe.site.values[(sess_numb-1)*wind_size: ]):
                new_sess['site' + str(n+1)] = freq_site_dict[site][0]
            #new_sess['user_id'] = userid
            ret_userid_list.append(userid)

            ret_data.iloc[index] = new_sess
            index += 1
        
    ret_csr = to_sitebow_csr(ret_data.values)
    
    return ret_csr, np.aaray(ret_userid_list)


# In[ ]:





# In[6]:


def to_sitebow_csr(inp_data):
    row  = []
    col  = []
    data = []
    for idx, elmnt in enumerate(inp_data):
        cnt = Counter(elmnt)        # for each string get {'site': how many in string}
        
        row += [idx] * len(cnt)     # row number in which value is parse
        data+= list(cnt.values())   # value
        col += list(cnt.keys())     # column number in which value is parse
        
    ret = csr_matrix((data, (row, col)), shape = (max(row)+1, max(col)+1))[:, 1:]
    
    return ret


# In[6]:


csr_data, y_s5_w3 = prepare_sparse_train_set_window(os.path.join(DATA, '3users'),
                                             os.path.join(MY_DATA, 'site_freq_3users.pkl'),
                                             5, 3)


# In[7]:


csr_data.todense()


# In[8]:


y_s5_w3


# In[ ]:





# In[7]:


def prepare_sparse_train_set_window_v2(path_to_csv, path_to_site_freq, sess_len, wind_size):
    
    #rows = 0
    #site_cnt = []
    ret_userid_list = []
    # form dataframe names ['site1, site2......site{session_length}']
    col_names = ['site' + str(n+1) for n in range(sess_len)]

    #pkl_name = 'site_freq_' + str(re.findall('\d+', path_to_data)[0]) + 'users.pkl'
    with open(path_to_site_freq, 'rb') as f:
        freq_site_dict = pickle.load(f)

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    # getting size of DataFrame
    # collect all sites ans them frequencies
    #for file_csv in (sorted(glob.glob(path_to_csv + '/*.csv'))):
    #    temp_dataframe = pd.read_csv(file_csv, usecols=['site'])
    #    site_cnt += list(temp_dataframe.site.values)
    #    rows += round(temp_dataframe.shape[0] / wind_size + 0.499)

    #site_cnt = Counter(site_cnt)

    # ceate DataFrame of known size
    #ret_data = pd.DataFrame(index = range(rows), columns = col_names)
    index = 0


    for file_csv in (sorted(glob.glob(path_to_csv + '/*.csv'))):
            temp_dataframe = pd.read_csv(file_csv, usecols=['site'])
            userid = int(re.findall('\d+', file_csv)[1])


            sess_numb = round(temp_dataframe.shape[0] / wind_size + 0.499)
            for idx in range(sess_numb - 1):
                #new dict for current session
                new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}

                sess_start = idx*wind_size
                sess_end   = idx*wind_size + sess_len

                for n, site in enumerate(temp_dataframe.site.values[sess_start : sess_end]):
                    new_sess['site' + str(n+1)] = freq_site_dict[site][0]
                #new_sess['user_id'] = userid
                ret_userid_list.append(userid)
                
                cnt_csr = Counter(new_sess.values())
                #print(len(cnt_csr))
                #print()
                #print(cnt_csr.values())
                #print()
                row_csr += [index] * len(cnt_csr)     # row number in which value is parse
                data_csr+= list(cnt_csr.values())   # value
                #print(cnt_csr.keys())
                #for site_n in cnt_csr.keys():
                col_csr += list(cnt_csr.keys())     # column number in which value is parse
                    #print(type(int(re.findall('\d+', site_n)[0])))
                    #col_csr += [int(re.findall('\d+', site_n)[0])]
                #put data in prepared DataFrame
                #ret_data.iloc[index] = new_sess
                index += 1


            new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}
            for n, site in enumerate(temp_dataframe.site.values[(sess_numb-1)*wind_size: ]):
                new_sess['site' + str(n+1)] = freq_site_dict[site][0]
            #new_sess['user_id'] = userid
            ret_userid_list.append(userid)
            cnt_csr = Counter(new_sess.values())
            row_csr += [index] * len(cnt_csr)     # row number in which value is parse
            data_csr+= list(cnt_csr.values())   # value
            #for site_n in cnt_csr.keys():
            col_csr += list(cnt_csr.keys())     # column number in which value is parse
            #    col_csr += [int(re.findall('\d+', site_n)[0])]
                

            #ret_data.iloc[index] = new_sess
            index += 1
        
    #ret_csr = to_sitebow_csr(ret_data.values)
   # 
   # 
   #     for idx, elmnt in enumerate(inp_data):
   #     cnt = Counter(elmnt)        # for each string get {'site': how many in string}
   #     
   #     row += [idx] * len(cnt)     # row number in which value is parse
   #     data+= list(cnt.values())   # value
   #     col += list(cnt.keys())     # column number in which value is parse#
    #print(data_csr)
    #print(col_csr)
    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    
    return ret_csr, np.array(ret_userid_list)


# In[8]:


csr_data_v2, y_s5_w3 = prepare_sparse_train_set_window_v2(os.path.join(DATA, '3users'),
                                             os.path.join(MY_DATA, 'site_freq_3users.pkl'),
                                             5, 3)


# In[9]:


csr_data_v2.toarray()


# In[ ]:





# In[10]:


get_ipython().run_cell_magic('time', '', "data_length = []\n\nfor n_users in [10, 150]:\n    for w_size, session_length in tqdm(itertools.product([10, 7, 5], [15, 10, 7, 5])):\n        if w_size <= session_length and (w_size, session_length) != (10, 10):\n            #print(w_size, session_length, n_users)\n            direct_name = str(n_users) + 'users'\n            pickle_name = 'site_freq_' + str(n_users) + 'users.pkl'\n            x, y = prepare_sparse_train_set_window_v2(os.path.join(DATA, direct_name),\n                                               os.path.join(MY_DATA, pickle_name),\n                                               session_length, \n                                               w_size, \n                                               )\n            data_length.append(x.shape[0])\n            #pkl_name_x = f'X_sparse_{n_users}users_s{session_length}_w{w_size}.pkl'\n            pkl_name_y = f'Y_sparse_{n_users}users_s{session_length}_w{w_size}.pkl'\n            \n            #with open(os.path.join(MY_DATA, pkl_name_x), 'wb') as x_pkl:\n            #    pickle.dump(x, x_pkl, protocol = 2)\n    \n            with open(os.path.join(MY_DATA, pkl_name_y), 'wb') as y_pkl:\n                pickle.dump(y, y_pkl, protocol = 2)\n            ")


# In[21]:


print(data_length)


# In[16]:


def write_answer_to_file(answer, file_address):
    if isinstance(answer, list) or isinstance(answer, np.ndarray):
        with open(os.path.join(ANSW, file_address), 'w') as out_f:
            for idx, elmnt in enumerate(answer):
                if idx == 0:
                    out_f.write(str(elmnt))
                else:
                    out_f.write(' ' + str(elmnt))
    else:
        with open(os.path.join(ANSW, file_address), 'w') as out_f:
            out_f.write(str(answer))


# In[17]:


write_answer_to_file(data_length, 'answer2_1.txt')


# In[ ]:





# # первичный анализ данных, проверка гипотез

# In[22]:


train_df = pd.read_csv(os.path.join(MY_DATA, 'train_data_10users.csv'),
                      index_col = 'session_id'
                      )


# In[23]:


train_df.head()


# In[24]:


train_df.info()


# In[25]:


train_df.user_id.value_counts()


# In[26]:


num_uniq_sites = [np.unique(train_df.values[i, :-1]).shape[0]
                 for i in range(train_df.shape[0])]

len(num_uniq_sites)


# In[27]:


pd.Series(num_uniq_sites).hist();


# проверим распределено ли кол-во уникальных сайтов в каждой сессии нормально

# In[28]:


sts.probplot(num_uniq_sites, plot=plt)


# In[29]:


fig = sm.qqplot(np.array(num_uniq_sites), sts.norm, line = 's')
plt.show()


# In[30]:


stat, p = sts.shapiro(num_uniq_sites)
print('Statistics=%.3f, p=%.8f' % (stat, p))


# In[31]:


alpha = 0.05
if p > alpha:
    answ2 = 'YES'
else:
    answ2 = 'NO'


# In[32]:


write_answer_to_file(answ2, 'answer2_2.txt')


# In[ ]:





# **Проверьте гипотезу о том, что пользователь хотя бы раз зайдет на сайт, который он уже ранее посетил в сессии из 10 сайтов. Давайте проверим с помощью биномиального критерия для доли, что доля случаев, когда пользователь повторно посетил какой-то сайт (то есть число уникальных сайтов в сессии < 10) велика: больше 95% (обратите внимание, что альтернатива тому, что доля равна 95% –  одностороняя).**

# In[33]:


has_two_similar = (np.array(num_uniq_sites) < 10).astype('int')
has_two_similar


# In[34]:


p_val = sts.binom_test(sum(has_two_similar), len(has_two_similar), p = 0.95, alternative='greater')
answ3 = p_val
answ3


# In[35]:


write_answer_to_file(answ3, 'answer2_3.txt')


# In[ ]:





# **Поcмотрим для этой доли 95% доверительный интервал Уилсона**

# In[36]:


wilson_int = proportion_confint(sum(has_two_similar), len(has_two_similar), method = 'wilson')
wilson_int


# In[37]:


answ4 = [round(wilson_int[0], 3), round(wilson_int[1], 3)]
answ4


# In[38]:


write_answer_to_file(answ4, 'answer2_4.txt')


# In[ ]:





# In[39]:


with open(os.path.join(MY_DATA, 'site_freq_10users.pkl'), 'rb') as f:
    freq_site_dict = pickle.load(f)


# **Постройте распределение частоты посещения сайтов (сколько раз тот или иной сайт попадается в выборке) для сайтов, которые были посещены как минимум 1000 раз.**
в спарс матрице первые н столбцов. кол-во 1, 2 и т.д.
предел? 
# In[43]:


with open(os.path.join(MY_DATA, 'site_freq_10users.pkl'), 'rb') as f:
    sites = pickle.load(f)


# In[45]:


len(sites)


# In[87]:


freq_sites = [el[1] for el in sites.values()]
freq_ind = [el[0] for el in sites.values()]


# In[ ]:





# **Постройте 95% доверительный интервал для средней частоты появления сайта в выборке (во всей, уже не только для тех сайтов, что были посещены как минимум 1000 раз) на основе bootstrap. Используйте столько же bootstrap-подвыборок, сколько сайтов оказалось в исходной выборке по 10 пользователям. Берите подвыборки из посчитанного списка частот посещений сайтов – не надо заново считать эти частоты. Учтите, что частоту появления нуля (сайт с индексом 0 появлялся там, где сессии были короче 10 сайтов) включать не надо. Округлите границы интервала до 3 знаков после запятой и запишите через пробел в файл *answer2_5.txt*. Это будет ответом на 5 вопрос теста.**

# In[172]:


def get_bootstrap_samples(data, n_samples, random_seed=17):
    np.random.seed(random_seed)
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


# In[173]:


def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, 
                 [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


# In[174]:


freq_data = pd.DataFrame({'freq': freq_sites}, index = freq_ind)


# предполагаем, что текущие данные лишь одна из возможных     
# выборок генеральной совокупности.     
# генерируем другие возможные выборки

# In[175]:


freq_samples = map(np.mean, get_bootstrap_samples(freq_data.freq.values, len(sites)))


# In[176]:


gg = stat_intervals(list(freq_samples), 0.05)


# In[177]:


gg


# In[178]:


answ5 = [round(gg[0], 3), round(gg[1], 3)]
answ5


# In[179]:


write_answer_to_file(answ5, 'answer2_5.txt')


# In[ ]:




