#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,sklearn,pandas,scipy,matplotlib,statsmodels -g')


# In[3]:


TQDM_DISABLE = True


# In[4]:


import warnings
warnings.filterwarnings("ignore")


# In[5]:


import time
time.strftime('%H:%M:%S')


# In[6]:


import os
import glob
import numpy as np
import pandas as pd

import re
#import random
import pickle

from tqdm.notebook import tqdm 

import itertools
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[7]:


get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:





# In[8]:


PATH_TO_DATA     = os.path.join('.', 'data')
PATH_TO_DATA_ALL = os.path.join('.', 'data', 'allcats')
PATH_TO_DATA_OWN = os.path.join('.', 'data_own')
PATH_TO_DATA_ALL_PREP = os.path.join('.', 'data', 'all_cats_prepared')
ANSW = os.path.join('.', 'answers')


# In[ ]:





# ### Подготовка данных.

# Получаем список всех пользователей. Он будет использоваться при формировании подвыборок.

# In[9]:


filenames_all = [re.findall('\d+', x)[0] for x in sorted(glob.glob(PATH_TO_DATA_ALL + '/*.csv'))]


# Первым делом предобработаем данные: получим длительность нахождения пользователя на сайте в секундах, зададим названия колонок/признаков.    
# Так же обращает на себя внимание тот факт, что есть некоторый лаг по времени в данных (у 92% пользователей). Например у пользователя 3388:

# 348    27-5-2014T16:01:53  
# 349    27-5-2014T16:04:49  
# 350    27-5-2014T14:42:15  
# 351    27-5-2014T14:42:17  

# В [оригинальной статье](http://ceur-ws.org/Vol-1703/paper12.pdf) отсутствует описание данного момента. В дополнение, дополнительно сформированные  
# директории с 10 и 150 пользователями так же не содержат данного лага. По этой причине просто устраним его.  
# Для устранения таких лагов отсортируем по времени и, выполнив вышеописанную предобработку, сохраним в новые файлы данные каждого пользователя.    
# Т.к. такую предобработку необходимо выполнять один раз, добавим проверку.

# In[10]:


get_ipython().run_cell_magic('time', '', "# 6 min\nif not os.path.isfile(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat0001.csv')):\n    not_monotonic = 0\n    for el in tqdm(filenames_all, desc = u'по всем пользователям', disable = TQDM_DISABLE):\n        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'cat{el}.csv'), header = None, sep = ';')#, parse_dates=[1]))\n        temp_dataframe.columns = ['target', 'timestamp', 'site']\n        temp_dataframe.timestamp = pd.to_datetime(temp_dataframe.timestamp)\n        if not temp_dataframe.timestamp.is_monotonic:\n            temp_dataframe.sort_values(by = 'timestamp', ascending = True, inplace = True, ignore_index = True)\n            not_monotonic += 1\n        temp_dataframe['time_diff'] = temp_dataframe.timestamp.diff().shift(-1).apply(lambda x: x.total_seconds())\n        temp_dataframe.loc[temp_dataframe.shape[0] - 1, 'time_diff'] = 0.0\n        \n        temp_dataframe.to_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{el}.csv'), index = False)\n        \n    print(not_monotonic / len(filenames_all))")


# In[ ]:





# In[11]:


def get_bootstrap_samples(data, n_samples, n_users = 10, random_seed=42):
    
    np.random.seed(random_seed)
    
    if n_users < 501:
        return np.random.randint(0, len(data), (n_samples, n_users))
        
    else:
        random_seed = 9999
        np.random.seed(random_seed)
        return np.random.randint(0, len(data), (1, n_users))


# In[12]:


def create_load_freq_site_dict(files, users_count, indexes = []):
    
    if users_count > 1000000:
        outp_freq_site_dict = pickle.load(open(os.path.join(PATH_TO_DATA_OWN, f'freq_site_dict_{users_count}users.pkl'), 'rb'))
    else:
        outp_freq_site_dict = dict()
        site_cnt = Counter()
        
        for idx in tqdm(indexes, desc = 'for all filenames', disable = TQDM_DISABLE):
            temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{files[idx]}.csv'), 
                                         usecols = ['site'])
            site_cnt += Counter(temp_dataframe.site.values)

        for idx, site_dict in enumerate(site_cnt.most_common(), start = 1):
            outp_freq_site_dict[site_dict[0]] = (idx, site_dict[1])    
    
    return outp_freq_site_dict


# Создаем словарь из всех сайтов с номером сайта, соответствующим его частоте. Это будут признаки.

# In[13]:


get_ipython().run_cell_magic('time', '', "# 16 sec\nif not os.path.isfile(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl')):\n    freq_site_dict = create_load_freq_site_dict(filenames_all, len(filenames_all), range(len(filenames_all)))\n    pickle.dump(freq_site_dict, open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl'), 'wb'))\n    print(len(freq_site_dict))\nelse:\n    freq_site_dict = pickle.load(open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl'), 'rb'))")


# In[ ]:





# Для начала нам необходимо определиться что будет считаться сессией. Все что у нас есть - это время захода на сайт и его url.    
# Таким образом за сессию будем считать последовательность из посещенных подряд сайтов одним пользователем.    
# Здесь возникает вопрос: каким количеством ограничить последовательность из сайтов? На первый взгляд чем длиннее последовательность, тем больше мы можем извлечь из нее информации и точнее определить пользователя.   
# При этом возникает ситуация, при которой в длинной сессии, определенной нами, оказываются сайты, посещенные за несколько дней. Здесь следует исходить из потребностей условного заказчика: в течении какого времени 
# ему необходимо определить пользователя, ведь в случае мошенника этот промежуток времени должен составлять секунды.
# Т.к. задача до таких подробностей для нас не конкретизирована, то предоставим условному заказчику обширные сведения для выбора. Тем более определить пользователя за секунды в нашем случае не выглядит возможным.

# ### Оценка охвата полных сессий при ограничении длительности сессии по времени.

# Для начала оценим охват полных сессий при выставлении ограничения. Оценку будем проводить по всей доступной нам выборке.

# In[14]:


def get_all_rollings_list(inp_flnames, inp_sess_len):
    ret_list = []
    for name in inp_flnames:
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{name}.csv'),
                                    usecols = ['time_diff'])
        ret_list += list(temp_dataframe['time_diff'].rolling(window = inp_sess_len, min_periods = 2).sum())[1:]
    
    return ret_list


# In[15]:


def get_area(inp_cntr, inp_borders, inp_list_len):
    ret_area = []
    for brd in inp_borders:
        ssum = 0
        for el in inp_cntr:
            if el < brd:
                ssum += inp_cntr[el]

        ret_area.append(ssum / inp_list_len)
        
    return ret_area


# Диапазон длин сессий и временных границ, внутри которых будем смотреть на результат.

# In[16]:


possible_sess_len = [5,  10, 15, 20]
possible_borders  = [10, 30, 60, 300, 600, 1800, 3600]


# In[17]:


get_ipython().run_cell_magic('time', '', "\nfig_coverage = make_subplots(rows=2, cols=2, shared_yaxes=True)\n\nfor idx, sessl in enumerate(tqdm(possible_sess_len, desc = 'по выбранным вариантам длины сессии', disable = TQDM_DISABLE)):\n    all_list = get_all_rollings_list(filenames_all, sessl)\n    cnt = Counter(all_list)\n    \n    area = get_area(cnt, possible_borders, len(all_list))\n    \n    fig_coverage.add_scatter( x = np.array(possible_borders), y = np.array(area), mode='lines',\n                             name = u'сессия длиною ' + str(sessl),\n                             row=int((idx/2)+1), col=int((idx%2)+1),\n                            )\n    one_more_trace=dict(\n                       type = 'scatter',\n                       x = [1800],\n                       y = [area[5]],\n                       mode = 'markers+text',\n                       text = str(area[5])[:6],\n                       marker = dict(\n                                  color = 'red',\n                                  size = 13,\n                                   ),\n                       showlegend = False,\n                       )\n    fig_coverage.append_trace(one_more_trace,\n                             row=int((idx/2)+1), col=int((idx%2)+1)\n                             )\n    fig_coverage.update_traces(textposition = 'top center') \n    ")


# In[18]:


fig_coverage.update_xaxes(type='category')
fig_coverage.update_layout(
                            autosize = False,
                            width  = 1200,
                            height = 800, 
                            title  = u'Охват данных в зависимости от ограничения по длительности сессии',
                            xaxis_title  = "Ограничение длительности сессии (c)",
                            xaxis2_title = "Ограничение длительности сессии (c)",
                            xaxis3_title = "Ограничение длительности сессии (c)",
                            xaxis4_title = "Ограничение длительности сессии (c)",
                            yaxis_title  = "Охват (%)",
                            yaxis2_title = "Охват (%)",
                            yaxis3_title = "Охват (%)",
                            yaxis4_title = "Охват (%)",
                            )
fig_coverage.show()


# Как мы видим при ограничении длины сессии в 1800 секунд (30 минут), только при ширине окна в 5 сайтов более 95% сессий будут полными (не содержат пустые значения при превышении лимита на ограничение по времени). При увеличении ширины окна до 20 сайтов, таких сессий будет уже только 85%, что уменьшает доступную информацию для классификации пользователя, но так же может служить и признаком.

# In[ ]:





# ### Дальше оценим влияние ширины окна и длины сессии в сайтах.

# In[19]:


variants_users_ws    = [3, 10, 100]
variants_window_size = [3, 5, 7, 10, 15, 20]
variants_sess_length = [3, 5, 7, 10, 15, 20]
map_dict = {3:0, 5:1, 7:2, 10:3, 15:4, 20:5}


# In[20]:


def prepare_sparse_train_set_window(inp_fnames, inp_idexes, inp_sess_len, 
                                    inp_wind_size, inp_freq_site_dict, inp_time_limit = 999999999):
    
    ret_userid_list = []
    # form dataframe names ['site1, site2......site{session_length}']
    col_names = ['site' + str(n+1) for n in range(inp_sess_len)]

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    row_index = 0
    early_exit = 0
    
    print_index = 15
    
    for idx in (inp_idexes):
        #print(row_index)
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{inp_fnames[idx]}.csv'), 
                                         usecols = ['site', 'time_diff']
                                        )
        userid = inp_fnames[idx]
              
        index = 0

        while index < temp_dataframe.shape[0]:
            condition_index = 0
            total_time = 0
            new_sess = {('site' + str(n+1)):0 for n in range(inp_sess_len)}
            # determine one current session
            while (condition_index < inp_sess_len) and (total_time < inp_time_limit) and ((index + condition_index) < temp_dataframe.shape[0]):
                new_sess['site' + str(condition_index + 1)] = inp_freq_site_dict[temp_dataframe.site[index + condition_index]][0]
                total_time += temp_dataframe.time_diff[index + condition_index]
                
                condition_index += 1

            if (condition_index <= inp_wind_size):
                index += (condition_index)
                early_exit += 1
            else:
                index += inp_wind_size

                
            cnt_csr  = Counter(new_sess.values())
            row_csr += [row_index] * len(cnt_csr)  # row number in which value is parse
            data_csr+= list(cnt_csr.values())      # value
            col_csr += list(cnt_csr.keys())        # column number in which value is parse
            ret_userid_list.append(userid)
            row_index += 1

    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    #print(early_exit)
    
    return ret_csr, np.array(ret_userid_list)


# In[21]:


get_ipython().run_cell_magic('time', '', "\nfull_acc_ws = []\nfull_f1_ws  = []\n\nfor nusers in variants_users_ws:\n    indexes = get_bootstrap_samples(filenames_all, 1, nusers)\n    acc_ws = np.zeros((len(variants_window_size), len(variants_sess_length)))\n    f1_ws  = np.zeros((len(variants_window_size), len(variants_sess_length)))\n    variant_ws_list = list(itertools.product(variants_window_size, variants_sess_length))\n\n    for window_size, session_len in tqdm(variant_ws_list, desc = str(nusers) + u' пользователя', disable = TQDM_DISABLE):\n            acc = 0\n            f1  = 0\n            if window_size <= session_len:\n                train_data_sparse, targets = prepare_sparse_train_set_window(filenames_all, indexes[0], session_len, \n                                                                             window_size, freq_site_dict)\n\n                try:\n                    X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                          test_size = 0.3, \n                                                          random_state = 821, stratify = targets)\n                except ValueError:\n                    X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                          test_size = 0.3, \n                                                          random_state = 821)\n                if len(set(y_train)) < 2: \n                    continue\n\n                clf = SGDClassifier(loss = 'log', n_jobs = -1, \n                                   max_iter = 1500,\n                                   )\n                clf.fit(X_train, y_train)\n                y_pred = clf.predict(X_valid)\n                acc = accuracy_score(y_valid, y_pred)\n                f1 = f1_score(y_valid, y_pred, average = 'macro')\n                \n\n            acc_ws[map_dict[window_size]][map_dict[session_len]] = acc    # row, column\n            f1_ws[map_dict[window_size]][map_dict[session_len]] = f1      # row, column\n    \n    full_acc_ws.append(acc_ws)\n    full_f1_ws.append(f1_ws)")


# In[22]:


fig_acc= make_subplots(rows=1, cols=3, #shared_xaxes=True, 
                       specs=[[{'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}]],
                       subplot_titles = [u'Для 3 пользователей', u'Для 10 пользователей', u'Для 100 пользователей']
                      )
for idx in range(len(variants_users_ws)):
    fig3da = go.Surface(    
                       #contours = {
                       #           "z": {"show": True, "start": 0.60, "end": 1, "size": 0.02}
                       #           },
                       z = full_acc_ws[idx], 
                       x = variants_window_size, 
                       y = variants_sess_length,  
                       showscale = False,
                       name = str(variants_users_ws[idx]) + u' users',
                       )
    fig_acc.add_trace( fig3da,
                      row=1, col=idx+1
                     )

fig_acc.update_layout(title  = u'Ожидаемая метрика accuracy в зависимости от ширины окна и длинны сессии',
                      height = 400,
                      margin = dict(l=30, r=30, b=0, t=70),
                      scene  = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'accuracy',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                  ),
                      scene2 = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'accuracy',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                   ),
                      scene3 = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'accuracy',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                   ),
                     )#update_layout
fig_acc.show()   


# In[23]:


fig_f1= make_subplots(rows=1, cols=3, #shared_xaxes=True, 
                        specs=[[{'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}]],
                        subplot_titles = [u'Для 3 пользователей', u'Для 10 пользователей', u'Для 100 пользователей']
                       )
for idx in range(len(variants_users_ws)):
    fig3da = go.Surface(    
                       #contours = {
                       #           "z": {"show": True, "start": 0.60, "end": 1, "size": 0.02}
                       #           },
                       z = full_f1_ws[idx], 
                       x = variants_window_size, 
                       y = variants_sess_length,  
                       showscale = False,
                       name = str(variants_users_ws[idx]) + u' users',
                       )
    fig_f1.add_trace( fig3da,
                      row=1, col=idx+1
                     )

fig_f1.update_layout(title  = u'Ожидаемая метрика f1-score в зависимости от ширины окна и длинны сессии',
                      height = 400,
                      margin = dict(l=30, r=30, b=0, t=70),
                      scene  = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'f1-score',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                  ),
                      scene2 = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'f1-score',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                   ),
                      scene3 = dict(xaxis_title = u'Длина сессии',
                        yaxis_title = u'Ширина окна',
                        zaxis_title = u'f1-score',
                        camera_eye  = dict(x=1.2, y=1.7, z=1.2),
                                   ),
                     )#update_layout
fig_f1.show()   


# Действительно, как мы видим, ожидаемые метрики accuracy и f1-score повышаются при увеличении ограничения по длительности сессии как по времени так и по количеству сайтов и при снижении размера окна между сессиями. С учетом этих знаний оценим возможные accuracy и f1-score с параметрами, близкими к наилучшим.

# <i>наилучшие параметры это размер окна порядка 1, но время расчетов на моих мощностях для него достаточно велико, будем считать для размера окна равного 5.    
# так же, чем больше длинна сессии, тем лучше, но мы остановимся на размере равным 20 опять же в виду вычислительных мощностей.

# In[ ]:





# ### Влияние количества пользователей на классификацию.

# Последний не рассмотренный нами параметр - количество пользователей (классов), для которых нам необходимо производить идентификацию.  
# По практике, чем больше классов, тем хуже классификация. Посмотрим.    
# 
# Для оценки классификации пользователей будем использовать следующее количество итераций:  
# 2 - 100, 5 - 100, 10 - 100, 50 - 50, 100 - 10, 500 - 5, далее по 1.  
# Оценку будем производить при помощи SGDClassifier, по причине его быстроты.

# In[ ]:





# Сформируем параметры для сбора статистик.

# In[24]:


variant_nusers = [ 2,   5,   10, 50, 100, 500, 1000, 2000, len(filenames_all)]
variant_naver  = [100, 100, 100, 50, 10,    5,   1,    1,    1]
len(variant_nusers) == len(variant_naver)


# Рассчитаем метрики по выбранным параметрам.

# In[25]:


get_ipython().run_cell_magic('time', '', "\nsess_len  = 20\nwind_size = 5\n\nfull_acc_users = []\nfull_f1_users  = []\n    \nparam_list = list(zip(variant_naver, variant_nusers))\nfor params in tqdm(param_list, desc = u'для всех вариантов пользователей и подвыборок', disable = TQDM_DISABLE):\n\n    indexes = get_bootstrap_samples(filenames_all, params[0], params[1])\n   \n    acc = []\n    f1  = []\n    for idx in indexes:\n        train_data_sparse, targets = prepare_sparse_train_set_window(filenames_all, idx, sess_len, \n                                                                     wind_size, freq_site_dict,\n                                                                    )\n        \n        try:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821, stratify = targets)\n        except ValueError:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821)\n        \n        if len(set(y_train)) < 2: \n            continue\n        \n        clf = SGDClassifier(loss = 'log', n_jobs = -1, \n                           max_iter = 1500,\n                           )\n        clf.fit(X_train, y_train)\n        y_pred = clf.predict(X_valid)\n        acc.append(accuracy_score(y_valid, y_pred))\n        f1.append(f1_score(y_valid, y_pred, average = 'macro'))\n        \n    full_acc_users.append(acc)\n    full_f1_users.append(f1)")


# Посмотрим на ожидаемые значения метрик accuracy и f1-score.

# In[26]:


fig_acc_users = go.Figure()
lin_acc = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_acc_users):
    lin_acc.append(np.mean(el))
    fig_acc_users.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_acc_users.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика accuracy в зависимости от количества пользователей для классификации',
    yaxis_title = u'accuracy',
    )

fig_acc_users.show()


# In[27]:


fig_f1_users = go.Figure()
lin_f1 = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_f1_users):
    lin_f1.append(np.mean(el))
    fig_f1_users.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_f1_users.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика f1-score в зависимости от количества пользователей для классификации',
    yaxis_title = u'f1-score',
    )

fig_f1_users.show()


# ### Посмотрим влияние добавления ограничения сессии по времени на метрики. 

# Зададим ограничение длительности сессии равным 600с (10 мин).

# In[28]:


get_ipython().run_cell_magic('time', '', "\nsess_len  = 20\nwind_size = 5\ntime_limit = 600\n\nfull_acc_tm_limit = []\nfull_f1_tm_limit  = []\n    \nparam_list = list(zip(variant_naver, variant_nusers))\nfor params in tqdm(param_list, desc = u'для всех вариантов пользователей и подвыборок', disable = TQDM_DISABLE):\n\n    indexes = get_bootstrap_samples(filenames_all, params[0], params[1])\n   \n    acc = []\n    f1  = []\n    for idx in indexes:\n        train_data_sparse, targets = prepare_sparse_train_set_window(filenames_all, idx, sess_len, \n                                                                     wind_size, freq_site_dict,\n                                                                     inp_time_limit = time_limit,\n                                                                    )\n        \n        try:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821, stratify = targets)\n        except ValueError:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821)\n        \n        if len(set(y_train)) < 2: \n            continue\n            \n        clf = SGDClassifier(loss = 'log', n_jobs = -1, \n                           max_iter = 1500,\n                           )\n        clf.fit(X_train, y_train)\n        y_pred = clf.predict(X_valid)\n        acc.append(accuracy_score(y_valid, y_pred))\n        f1.append(f1_score(y_valid, y_pred, average = 'macro'))\n        \n    full_acc_tm_limit.append(acc)\n    full_f1_tm_limit.append(f1)")


# In[29]:


fig_acc_tm_limit = go.Figure()
lin_acc = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_acc_tm_limit):
    lin_acc.append(np.mean(el))
    fig_acc_tm_limit.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_acc_tm_limit.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика accuracy в зависимости от ограничения по времени и количества пользователей для классификации',
    yaxis_title = u'accuracy',
    )

fig_acc_tm_limit.show()


# In[30]:


fig_f1_tm_limit = go.Figure()
lin_f1 = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_f1_tm_limit):
    lin_f1.append(np.mean(el))
    fig_f1_tm_limit.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_f1_tm_limit.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика f1-score в зависимости от ограничения по времени и количества пользователей для классификации',
    yaxis_title = u'f1-score',
    )

fig_f1_tm_limit.show()


# Ожидаемо метрики снижаются. При ограничении в 600с, как мы видели ранее, полными остаются только 77,3% сессий при выбранной нами длине сессии.  
# При уменьшении ограничения по длительности сессии больше сессий станут не полными, отчего еще больше информации будет потяряно, а, следовательно,  
# уменьшатся и значения метрик.

# На всех полученных графиках значения метрик для большого количества пользователей малы и классификация для них не представляется сколь нибудь обоснованной.  
# В дальнейшем будем производить оценку для сокращенного количества вариантов пользователей: 2, 5, 10, 50, 100.

# In[31]:


variant_nusers = [ 2,   5,   10, 50, 100]
variant_naver  = [100, 100, 100, 50, 10 ]
len(variant_nusers) == len(variant_naver)


# Попробуем добавить временные метки посещения сайтов. Идея заключается в том, что сейчас у нас не используется информация о продолжительности посещения сайтов и о порядке их посещения.  
# Включение порядка через n-граммы приведет к кратному увеличению размерности признакового пространства, длительность посещения сайтов только к двукратному.  
# Посмотрим на влияние таких признаков, а так же признаков времени начала сессии и дня начала сессии.  

# In[32]:


WEEK = 7


# In[33]:


def prepare_sparse_train_set_total(inp_fnames, inp_index, inp_sess_len, inp_wind_size, 
                                   inp_freq_site_dict, inp_time_limit = 999999999):
    
    ret_userid_list = []

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    row_index = 0
    #early_exit = 0
    
    #print_index = 15
    time_start_idx = len(inp_freq_site_dict)
    
    #for idx in tqdm(range(len(inp_fnames)), disable = TQDM_DISABLE):
    for idx in tqdm(inp_index, disable = TQDM_DISABLE):
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{inp_fnames[idx]}.csv'), 
                                         usecols = ['timestamp', 'site', 'time_diff'],
                                         parse_dates=['timestamp']
                                    )
        userid = inp_fnames[idx]
              
        index = 0

        while index < temp_dataframe.shape[0]:
            condition_index = 0
            total_time = 0
            
            new_sess = defaultdict(float)
            #add 'dow' and 'hour' as OHE
            dow  = temp_dataframe.timestamp[index].dayofweek
            hour = temp_dataframe.timestamp[index].hour
                
            # all sites + all site times / + 7 days (for hour)
            new_sess[time_start_idx*2 + dow + 1] = 1
            new_sess[time_start_idx*2 + WEEK + hour - 7 + 1] = 1
            
            # determine one current session
            while (condition_index < inp_sess_len) and (total_time < inp_time_limit) and ((index + condition_index) < temp_dataframe.shape[0]):
                site_time = temp_dataframe.time_diff[index + condition_index]
                if site_time == 0:
                    site_time += 1
                        
                # normalize on ~[0, 1)
                new_sess[inp_freq_site_dict[temp_dataframe.site[index + condition_index]][0]] += (1/10)
                new_sess[inp_freq_site_dict[temp_dataframe.site[index + condition_index]][0] + time_start_idx] += (site_time/inp_time_limit)
                total_time += temp_dataframe.time_diff[index + condition_index]
                
                condition_index += 1

            if (condition_index <= inp_wind_size):
                index += (condition_index)
                #early_exit += 1
            else:
                index += inp_wind_size

                
            row_csr += [row_index] * len(new_sess)   # row number in which value is parse
            data_csr+= list(new_sess.values())       # value
            col_csr += list(new_sess.keys())         # column number in which value is parse

            ret_userid_list.append(userid)
            row_index += 1

    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    
    return ret_csr, np.array(ret_userid_list)


# In[34]:


get_ipython().run_cell_magic('time', '', "\nsess_len  = 20\nwind_size = 5\ntime_limit = 600\n\nfull_total_acc = []\nfull_total_f1  = []\n    \nparam_list = list(zip(variant_naver, variant_nusers))\nfor params in tqdm(param_list, desc = u'для всех вариантов пользователей и подвыборок', disable = TQDM_DISABLE):\n\n    indexes = get_bootstrap_samples(filenames_all, params[0], params[1])\n   \n    acc = []\n    f1  = []\n    for idx in indexes:\n        train_data_sparse, targets = prepare_sparse_train_set_total(filenames_all, idx, sess_len, \n                                                                     wind_size, freq_site_dict,\n                                                                     inp_time_limit = time_limit,\n                                                                    )\n        \n        try:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821, stratify = targets)\n        except ValueError:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size = 0.3, \n                                                      random_state = 821)\n        \n        if len(set(y_train)) < 2: \n            continue\n            \n        clf = SGDClassifier(loss = 'log', n_jobs = -1, \n                            max_iter = 1500,\n                            #eta0 = 1.0, learning_rate = 'adaptive',\n                            #penalty = 'l1',\n                           )\n        clf.fit(X_train, y_train)\n        y_pred = clf.predict(X_valid)\n        acc.append(accuracy_score(y_valid, y_pred))\n        f1.append(f1_score(y_valid, y_pred, average = 'macro'))\n        \n    full_total_acc.append(acc)\n    full_total_f1.append(f1)")


# In[35]:


fig_total_acc = go.Figure()
lin_acc = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_total_acc):
    lin_acc.append(np.mean(el))
    fig_total_acc.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_total_acc.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика accuracy в зависимости от количества пользователей для классификации',
    yaxis_title = u'accuracy',
    )

fig_total_acc.show()


# In[36]:


fig_total_f1 = go.Figure()
lin_f1 = []
x_names = [str(el) + u' пользователей' for el in variant_nusers]
x_names[0] = str(variant_nusers[0]) + u' пользователя'
for idx, el in enumerate(full_total_f1):
    lin_f1.append(np.mean(el))
    fig_total_f1.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig_total_f1.update_layout(
    autosize = False,
    width  = 900,
    height = 500, 
    title  = u'Ожидаемая метрика f1-score в зависимости от количества пользователей для классификации',
    yaxis_title = u'f1-score',
    )

fig_total_f1.show()


# Полученный результат контринтуитивенЖ мы добавили признаков, а метрики снизились.  
# Вероятно сказалось то, что пространство признаков у нас более 115000, при простанстве объектов не проевышающем 21000.  
# Так же стоит учитывать стохастическую природу SGDClassifier.  

# ### Выводы:
# В первой части была рассмотрена возможность идентификации пользователей по последовательности посещенных ими сайтов.  
# Рассмотрено влияние гиперпараметров и размерности признакового пространства.  
# 
# Дальнейшая попытка улучшения метрик для <u>всех</u> возможных случаев обречена на провал.  
# Что бы не блуждать в темноте специфицируем задачу и попробуем получить хорошие метрики для нее.  

# <i>next in part 2

# In[ ]:





# In[37]:


time.strftime('%H:%M:%S')


# In[ ]:




