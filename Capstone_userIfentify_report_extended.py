#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[3]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,sklearn,pandas,scipy,matplotlib,statsmodels -g')


# TODO:
#     
# - [x] Оценка соответствия длины сессии в сайтах и соответствующей ей длины сессии в секундах
# - [ ] ?ограничение длинны сессии по времени?
# - [ ] ?pipeline?
# - [ ] генерация новых признаков
# - [ ] отбор признаков
# - [ ] выбор алгоритма
# - [ ] оптимизация параметров алгоритма
# - [ ] выводы
# - [ ] оформление (текст, описание)

# In[4]:


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
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[5]:


get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:





# In[6]:


PATH_TO_DATA     = os.path.join('.', 'data')
PATH_TO_DATA_ALL = os.path.join('.', 'data', 'allcats')
PATH_TO_DATA_OWN = os.path.join('.', 'data_own')
PATH_TO_DATA_ALL_PREP = os.path.join('.', 'data', 'all_cats_prepared')
ANSW = os.path.join('.', 'answers')


# In[ ]:





# ### Подготовка данных

# Получаем список всех пользователей. Он будет использоваться при формировании подвыборок.

# In[7]:


filenames_all = [re.findall('\d+', x)[0] for x in sorted(glob.glob(PATH_TO_DATA_ALL + '/*.csv'))]
#filenames_10users_sub = [re.findall('\d+', x[-12:])[0] for x in sorted(glob.glob(os.path.join(PATH_TO_DATA, '10users') + '/*.csv'))]
#filenames_150users_sub = [re.findall('\d+', x[-12:])[0] for x in sorted(glob.glob(os.path.join(PATH_TO_DATA, '150users') + '/*.csv'))]


# Первым делом предобработаем данные: получим длительность нахождения пользователя на сайте в секундах, зададим названия колонок/признаков.    
# Так же обращает на себя внимание тот факт, что есть некоторый лаг по времени в данных (у 92% пользователей). Например у пользователя 3388:
348    27-5-2014T16:01:53
349    27-5-2014T16:04:49
350    27-5-2014T14:42:15
351    27-5-2014T14:42:17
# Для устранения таких лагов отсортируем по времени и, выполнив вышеописанную предобработку, сохраним в новые файлы данные каждого пользователя.    
# Т.к. такую предобработку необходимо выполнять один раз, добавим проверку.

# In[8]:


get_ipython().run_cell_magic('time', '', "# 6 min\nif not os.path.isfile(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat0001.csv')):\n    not_monotonic = 0\n    for el in tqdm(filenames_all, desc = 'for all filenames'):\n        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'cat{el}.csv'), header = None, sep= ';')#, parse_dates=[1]))\n        temp_dataframe.columns = ['target', 'timestamp', 'site']\n        temp_dataframe.timestamp = pd.to_datetime(temp_dataframe.timestamp)\n        if not temp_dataframe.timestamp.is_monotonic:\n            temp_dataframe.sort_values(by = 'timestamp', ascending = True, inplace = True, ignore_index = True)\n            not_monotonic += 1\n        temp_dataframe['time_diff'] = temp_dataframe.timestamp.diff().shift(-1).apply(lambda x: x.total_seconds())\n        temp_dataframe.loc[temp_dataframe.shape[0] - 1, 'time_diff'] = 0.0\n        \n        temp_dataframe.to_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{el}.csv'), index = False)\n        \n    print(not_monotonic / len(filenames_all))")

temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'cat{filenames_all[0]}.csv'), header = None, sep= ';')#, parse_dates=[1]))
temp_dataframe.columns = ['target', 'timestamp', 'site']
temp_dataframe.timestamp = pd.to_datetime(temp_dataframe.timestamp)
if not temp_dataframe.timestamp.is_monotonic:
    temp_dataframe.sort_values(by = 'timestamp', ascending = True, inplace = True, ignore_index = True)
    #not_monotonic += 1
temp_dataframe['time_diff'] = temp_dataframe.timestamp.diff().shift(-1).apply(lambda x: x.total_seconds())
# In[ ]:





# In[9]:


#Подготовим данные: разделим каждый файл на 3 колонки и пересохраним, что бы в дальнейшем читать только необходимые нам данные


# In[10]:


def get_bootstrap_samples(data, n_samples, n_users = 10, random_seed=42):
    
    np.random.seed(random_seed)
    
    if n_users < 501:
        return np.random.randint(0, len(data), (n_samples, n_users))
        
    #elif n_users < 3000:    
    else:
        random_seed = 9999
        np.random.seed(random_seed)
        return np.random.randint(0, len(data), (1, n_users))
    #else:
    #    return np.ndarray()


# In[11]:


def create_load_freq_site_dict(files, users_count, indexes = []):
    
    if users_count > 1000000:
        outp_freq_site_dict = pickle.load(open(os.path.join(PATH_TO_DATA_OWN, f'freq_site_dict_{users_count}users.pkl'), 'rb'))
    else:
        outp_freq_site_dict = dict()
        site_cnt = Counter()
        
        for idx in tqdm(indexes, desc = 'for all filenames'):
            #temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'cat{files[idx]}.csv'), header = None, sep= ';')#, parse_dates=[1]))
            #temp_dataframe.columns = ['target', 'timestamp', 'site']
            temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{files[idx]}.csv'), 
                                         #sep= ';', 
                                         usecols = ['site'])#, parse_dates=[1]))
            site_cnt += Counter(temp_dataframe.site.values)
            #rows += round(temp_dataframe.shape[0] / sess_len + 0.499)

        #rows = rows

        for idx, site_dict in enumerate(site_cnt.most_common(), start = 1):
            outp_freq_site_dict[site_dict[0]] = (idx, site_dict[1])    
    
    return outp_freq_site_dict


# Создаем словарь из всех сайтов с номером сайта, соответствующим его частоте. Это будут признаки.

# In[12]:


get_ipython().run_cell_magic('time', '', "# 16 sec\nif not os.path.isfile(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl')):\n    freq_site_dict = create_load_freq_site_dict(filenames_all, len(filenames_all), range(len(filenames_all)))\n    pickle.dump(freq_site_dict, open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl'), 'wb'))\n    print(len(freq_site_dict))\nelse:\n    #pkl_name = os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl')\n    freq_site_dict = pickle.load(open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl'), 'rb'))")


# Для начала нам необходимо определиться что будет считаться сессией. Все что у нас есть - это время захода на сайт и его url.    
# Таким образом за сессию будем считать последовательность из посещенных подряд сайтов одним пользователем.    
# Здесь возникает вопрос: каким количеством ограничить последовательность из сайтов? На первый взгляд чем длиннее последовательность, тем больше мы можем извлечь из нее информации и точнее определить пользователя.   
# При этом возникает ситуация, при которой в длинной сессии, определенной нами, оказываются сайты, посещенные за несколько дней. Здесь следует исходить из потребнойтей условного заказчика: в течении какого времени 
# ему необходимо определить пользователся, ведь в случае мошенника этот промежуток времени должен составлять секунды.
# Т.к. задача до таких подробностей для нас не конкретизирована, то предоставим условному заказчику обширные сведения для выбора. Тем более определить пользователя за секунды в нашем случае не выглядит возможным.

# ### Оценка охвата при ограничении длительности сессии по времени

# Для начала оценим охват полных сессий при выставлении ограничения. Оценку будем проводить по всей доступной нам выборке.

# In[13]:


def get_all_rollings_list(inp_flnames, inp_sess_len):
    ret_list = []
    for name in inp_flnames:
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{name}.csv'),
                                    usecols = ['time_diff'])
        ret_list += list(temp_dataframe['time_diff'].rolling(window = inp_sess_len, min_periods = 2).sum())[1:]
    
    return ret_list


# In[14]:


def get_area(inp_cntr, inp_borders, inp_list_len):
    ret_area = []
    for brd in inp_borders:
        ssum = 0
        for el in inp_cntr:
            if el < brd:
                ssum += inp_cntr[el]

        #ret_area.append(int( 100 * ssum / inp_list_len))
        ret_area.append(ssum / inp_list_len)
        
    return ret_area


# In[ ]:





# In[36]:


get_ipython().run_cell_magic('time', '', "possible_sess_len = [5,  10, 15, 20]\npossible_borders  = [10, 30, 60, 300, 600, 1800, 3600]\n\nfig_coverage = make_subplots(rows=2, cols=2, shared_yaxes=True)\n#fig_len = make_subplots(rows=2, cols=2, shared_yaxes=True)\n\nfor idx, sessl in enumerate(tqdm(possible_sess_len, desc = 'session length variants')):\n    all_list = get_all_rollings_list(filenames_all, sessl)\n    cnt = Counter(all_list)\n    \n    area = get_area(cnt, possible_borders, len(all_list))\n    \n    fig_coverage.add_scatter( x=np.array(possible_borders), y=np.array(area), mode='lines',\n                             name=u'сессия длиною ' + str(sessl),\n                             row=int((idx/2)+1), col=int((idx%2)+1),\n                            )\n    one_more_trace=dict(\n                       type='scatter',\n                       x=[1800],\n                       y=[area[5]],\n                       mode='markers+text',\n                       text = str(area[5])[:6],\n                       marker=dict(\n                                  color='red',\n                                  size=13,\n                                   ),\n                       showlegend=False,\n                       )\n    fig_coverage.append_trace(one_more_trace,\n                             row=int((idx/2)+1), col=int((idx%2)+1)\n                             )\n    fig_coverage.update_traces(textposition='top center') \n    \n    \n    #fig_len.add_trace(go.Box(y=all_list, name = sessl),\n    #                         row=int((idx/2)+1), col=int((idx%2)+1)\n    #                         )")


# In[15]:


fig_coverage.update_xaxes(type='category')
fig_coverage.update_layout(
                            autosize=False,
                            width=1200,
                            height=800, 
                            title = u'Охват данных в зависимости от ограничения по длительности сессии',
                            xaxis_title ="Ограничение длительности сессии (c)",
                            xaxis2_title="Ограничение длительности сессии (c)",
                            xaxis3_title="Ограничение длительности сессии (c)",
                            xaxis4_title="Ограничение длительности сессии (c)",
                            yaxis_title ="Охват (%)",
                            yaxis2_title="Охват (%)",
                            yaxis3_title="Охват (%)",
                            yaxis4_title="Охват (%)",
                            )
fig_coverage.show()


# Как мы видим при ограничении длины сессии в 1800 секунд (30 минут), только при ширине окна в 5 сайтов более 95% сессий будут полными (не содержат пустые значения при превышении лимита на ограничение по времени). При увеличении ширины окна до 20 сайтов, таких сессий будет уже только 85%, что уменьшает доступную информацию для классификации пользователя, но так же может служить и признаком.

# In[ ]:





# ### Дальше оценим влияние ширины окна и длины сессии в сайтах

# In[15]:


variants_users_ws    = [3, 10, 100]
variants_window_size = [3, 5, 7, 10, 15, 20]
variants_sess_length = [3, 5, 7, 10, 15, 20]
map_dict = {3:0, 5:1, 7:2, 10:3, 15:4, 20:5}


# In[37]:


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
    for idx in (inp_idexes):
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'cat{inp_fnames[idx]}.csv'), 
                                         usecols = ['site', 'time_diff']
                                        )
        userid = inp_fnames[idx]
        '''
            sess_numb = round(temp_dataframe.shape[0] / inp_wind_size + 0.499)
            for idx in range(sess_numb - 1):
                #new dict for current session
                new_sess = {('site' + str(n+1)):0 for n in range(inp_sess_len)}

                sess_start = idx*inp_wind_size
                sess_end   = idx*inp_wind_size + inp_sess_len

                for n, site in enumerate(temp_dataframe.site.values[sess_start : sess_end]):
                    new_sess['site' + str(n+1)] = inp_freq_site_dict[site][0]
                #new_sess['user_id'] = userid
                ret_userid_list.append(userid)
        '''                
        index = 0

        while index < temp_dataframe.shape[0]:
            condition_index = 0
            total_time = 0
            new_sess = {('site' + str(n+1)):0 for n in range(inp_sess_len)}
            # determine one current session
            while (condition_index < inp_sess_len) and (total_time < inp_time_limit):
                new_sess['site' + str(condition_index+1)] = inp_freq_site_dict[temp_dataframe.site[index]][0]
                total_time += temp_dataframe.time_diff[index]

                condition_index += 1

            if (condition_index == inp_sess_len):
                index += inp_wind_size
            else: #if condition_index < inp_wind_size:   # otherwise we losing sites in data
                index += condition_index
                early_exit += 1

                #row_index += 1

            cnt_csr = Counter(new_sess.values())
            #row_csr += [index] * len(cnt_csr)     # row number in which value is parse
            row_csr += [row_index] * len(cnt_csr)  # row number in which value is parse
            data_csr+= list(cnt_csr.values())      # value
            col_csr += list(cnt_csr.keys())        # column number in which value is parse
            ret_userid_list.append(userid)
            row_index += 1

            '''
            new_sess = {('site' + str(n+1)):0 for n in range(inp_sess_len)}
            for n, site in enumerate(temp_dataframe.site.values[(sess_numb-1)*inp_wind_size: ]):
                new_sess['site' + str(n+1)] = inp_freq_site_dict[site][0]
            ret_userid_list.append(userid)
            cnt_csr = Counter(new_sess.values())
            row_csr += [index] * len(cnt_csr)     # row number in which value is parse
            data_csr+= list(cnt_csr.values())   # value
            col_csr += list(cnt_csr.keys())     # column number in which value is parse
                
            index += 1
            '''
    
    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    #print(early_exit)
    
    return ret_csr, np.array(ret_userid_list)


# In[38]:


get_ipython().run_cell_magic('time', '', "\nfull_acc_ws = []\n\nfor nusers in variants_users_ws:\n    indexes = get_bootstrap_samples(filenames_all, 1, nusers)\n    acc_ws = np.zeros((len(variants_window_size), len(variants_sess_length)))\n    variant_ws_list = list(itertools.product(variants_window_size, variants_sess_length))\n\n    for window_size, session_len in tqdm(variant_ws_list, desc = str(nusers) + ' users'):\n            acc = 0\n            if window_size <= session_len:\n                train_data_sparse, targets = prepare_sparse_train_set_window(filenames_all, indexes[0], session_len, \n                                                                             window_size, freq_site_dict)\n\n                try:\n                    X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                          test_size=0.3, \n                                                          random_state=821, stratify=targets)\n                except ValueError:\n                    X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                          test_size=0.3, \n                                                          random_state=821)\n                    if len(set(y_train)) == 1:\n                        continue\n\n                clf = SGDClassifier(loss = 'log', n_jobs = -1, )\n                clf.fit(X_train, y_train)\n                y_pred = clf.predict(X_valid)\n                acc = accuracy_score(y_valid, y_pred)\n\n            acc_ws[map_dict[window_size]][map_dict[session_len]] = acc    # row, column\n    \n    full_acc_ws.append(acc_ws)")


# In[39]:


fig = make_subplots(rows=1, cols=3, #shared_xaxes=True, 
                    specs=[[{'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}, {'type': 'surface', 'is_3d': True}]],
                    subplot_titles=[u'Для 3 пользователей', u'Для 10 пользователей', u'Для 100 пользователей']
                   )
for idx in range(len(variants_users_ws)):
    fig3da = go.Surface(    
                       #contours = {
                       #           "z": {"show": True, "start": 0.60, "end": 1, "size": 0.02}
                       #           },
                       z = full_acc_ws[idx], 
                       x = variants_window_size, 
                       y = variants_sess_length,  
                       showscale=False,
                       name = str(variants_users_ws[idx]) + u' users',
                       # colorbar_z = 1
                       )
    fig.add_trace( fig3da,
                    row=1, col=idx+1
                   )
    #fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                              highlightcolor="limegreen", project_z=True))

fig.update_layout(title = u'Ожидаемая точность в зависимости от ширины окна и длинны сессии',
                  height = 400,
                  margin=dict(l=30, r=30, b=0, t=70),
                  scene = dict(xaxis_title=u'Длина сессии',
                    yaxis_title=u'Ширина окна',
                    zaxis_title=u'Точность',
                    camera_eye=dict(x=1.2, y=1.7, z=1.2),
                              ),
                  scene2 = dict(xaxis_title=u'Длина сессии',
                    yaxis_title=u'Ширина окна',
                    zaxis_title=u'Точность',
                    camera_eye=dict(x=1.2, y=1.7, z=1.2),
                               ),
                  scene3 = dict(xaxis_title=u'Длина сессии',
                    yaxis_title=u'Ширина окна',
                    zaxis_title=u'Точность',
                    camera_eye=dict(x=1.2, y=1.7, z=1.2),
                               ),
                 )#update_layout
fig.show()   


# ## <i>Вот здесь постигла неожиданность. После того как мы отсортировали данные для устранения лага, точность понизилась и, дополнительно, появилась сильная вариативность в параметрах. Без сортировки точность уверенно росла при уменьшении ширины окна и увеличении длины сессии. В данном расчете видим, что совсем не так. Возвращаемся к работе с неотсортированными данными.

# ~~Действительно, как мы видим, ожидаемая точность повышается при увеличении ограничения по длительности сессии как по времени так и по количеству сайтов и при снижении окна между сессиями.~

# In[ ]:





# In[78]:


#list(temp_dataframe['time_diff'].rolling(window = 5, min_periods = 2))


# In[ ]:


def calculate_number_of_sessions(inp_df, inp_time_len, inp_sess_len):
    ret_sess_numb = 0
    idx = 0
    while idx < (inp_df.shape[0] - ):
        
    
    return ret_sess_numb


# ### Влияние количества пользователей на классификацию

# Последний не рассмотренный нами параметр - количество пользователей (классов), для которых нам необходимо производить идентификацию. По практике, чем больше классов, тем хуже классификация. Посмотрим.    
# Будем производить оценку для классификации выборок из 2, 5, 10, 50, 100, 500, 1000, 2000, 3370 пользователей.
# 
# т.о. для оценки классификации с диапазоном ХХХ для 2х пользователей понадобится ХХХ итераций, 5 - , 10 - , 50 - , 100 - , 500 - , 1000 - , 2000 - , 3370 - 
# Оценку будем произвлдить при помощи SGDClassifier, по причине его быстроты.

# In[20]:





# Сформируем параметры для сбора статистик

# In[27]:


variant_nusers = [ 2,   5,   10, 50, 100, 500, 1000, 2000, len(filenames_all)]
variant_naver  = [100, 100, 100, 50, 10,   5,   1,    1,    1]
len(variant_nusers) == len(variant_naver)


# In[28]:


#for params in tqdm(zip(variant_naver, variant_nusers)):
#    print(params[0], params[1])


# Рассчитаем точность по выбранным параметрам

# <i>как мы видели ранее, оптимальным значением для размера окна является 1 (при этом стоит обратить внимание не переобучается ли модель в таких случаях), но, исходя     
# из соотношения затраченного времени / выгоды для учебного проекта, все дальнейшие расчеты будем вести все же с параметром размера окна равным 5. т.к. для параметра 1 
# расчеты на моем железе производятся неоправдано долго

# In[29]:


get_ipython().run_cell_magic('time', '', "\nsess_len  = 20\nwind_size = 5\n\nfull_acc = []\n    \nparam_list = list(zip(variant_naver, variant_nusers))\nfor params in tqdm(param_list, desc = 'for all variants of user and subsample'):\n\n    indexes = get_bootstrap_samples(filenames_all, params[0], params[1])\n   \n    acc = []\n    for idx in indexes:\n        train_data_sparse, targets = prepare_sparse_train_set_window(filenames_all, idx, sess_len, \n                                                                     wind_size, freq_site_dict,\n                                                                    )\n        \n        try:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size=0.3, \n                                                      random_state=821, stratify=targets)\n        except ValueError:\n            X_train, X_valid, y_train, y_valid = train_test_split(train_data_sparse, targets, \n                                                      test_size=0.3, \n                                                      random_state=821)\n            if len(set(y_train)) == 1:\n                #print('gg')\n                continue\n\n        \n        clf = SGDClassifier(loss = 'log', n_jobs = -1, )\n        clf.fit(X_train, y_train)\n        y_pred = clf.predict(X_valid)\n        acc.append(accuracy_score(y_valid, y_pred))\n        \n    full_acc.append(acc)")


# Посмотрим на ожидаемую точность (accuracy)

# In[30]:


fig = go.Figure()
lin_acc = []
x_names = [str(el) + u'users' for el in variant_nusers]
for idx, el in enumerate(full_acc):
    lin_acc.append(np.mean(el))
    fig.add_trace(go.Box(y=el, name = x_names[idx]))
    
#fig.add_trace(go.Scatter(y=lin_acc, x = x_names))
fig.update_layout(
    autosize=False,
    width=900,
    height=500, 
    title = u'Ожидаемая точность в зависимости от количества пользователей для классификации')

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




