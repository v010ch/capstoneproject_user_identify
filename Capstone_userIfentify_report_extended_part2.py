#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,sklearn,pandas,scipy,matplotlib,statsmodels -g')


# In[3]:


TQDM_DISABLE = True


# ### Предположим нам специфицировали задачу.    
# ### Требуется производить идентификацию для пользователей из директории '150users' и ограничением по времени не превышающим 10 минут.

# In[4]:


import os
import time
import glob
import numpy as np
import pandas as pd

import re
#import random
import pickle
import gc

from tqdm.notebook import tqdm 

import itertools
from collections import Counter
from collections import defaultdict
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split, StratifiedKFold#, GridSearchCV
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, f1_score

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import xgboost as xgb


# In[6]:


get_ipython().run_line_magic('pylab', 'inline')


# In[7]:


from sklearn import feature_selection


# Подготовим окружение, данные, известные параметры

# In[8]:


PATH_TO_DATA     = os.path.join('.', 'data')
PATH_TO_DATA_ALL = os.path.join('.', 'data', '150users')
PATH_TO_DATA_OWN = os.path.join('.', 'data_own')
PATH_TO_DATA_ALL_PREP = os.path.join('.', 'data', '150users_prepared')
ANSW = os.path.join('.', 'answers')


# In[9]:


filenames_all = [re.findall('\d+', x)[-1] for x in sorted(glob.glob(PATH_TO_DATA_ALL + '/*.csv'))]


# In[10]:


WEEK = 7


# In[ ]:





# In[11]:


time.strftime('%H:%M:%S')


# In[12]:


get_ipython().run_cell_magic('time', '', "# 6 min\nif not os.path.isfile(os.path.join(PATH_TO_DATA_ALL_PREP, f'user0006.csv')):\n    not_monotonic = 0\n    for el in tqdm(filenames_all, desc = u'по всем пользователям', disable = TQDM_DISABLE):\n        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'user{el}.csv'), sep= ',')#, parse_dates=[1]))\n        #temp_dataframe.columns = ['target', 'timestamp', 'site']\n        temp_dataframe.columns = ['timestamp', 'site']\n        temp_dataframe.timestamp = pd.to_datetime(temp_dataframe.timestamp)\n        if not temp_dataframe.timestamp.is_monotonic:\n            temp_dataframe.sort_values(by = 'timestamp', ascending = True, inplace = True, ignore_index = True)\n            not_monotonic += 1\n        temp_dataframe['time_diff'] = temp_dataframe.timestamp.diff().shift(-1).apply(lambda x: x.total_seconds())\n        temp_dataframe.loc[temp_dataframe.shape[0] - 1, 'time_diff'] = 0.0\n        temp_dataframe['target'] = int(el)\n        \n        temp_dataframe.to_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'user{el}.csv'), index = False)\n        \n    print(not_monotonic / len(filenames_all))")


# Длина сессии - чем больше, тем лучше. Ограничимся 20 в виду вычислительных возможностей.    
# Размер окна - чем меньше, тем лучше. Ограничимся 5 в виду вычислительных возможностей.    
# Временной лимит сессии нам задан - 10 минут = 600 секунд.

# In[13]:


sess_len  = 20 #30
wind_size = 5
time_limit = 600


# Зафиксируем random_seed для воспроизведения результатов.    
# Все оценки будем производить через StratifiedKFold. Зададим его.

# In[14]:


rand_seed = 52
skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 17)


# Словарь будем использовать только по этим пользователям. В противном случае получаем линейно зависимое пространство признаков слишком большой размерности.

# In[15]:


def create_load_freq_site_dict(files):
    
    outp_freq_site_dict = dict()
    site_cnt = Counter()

    for idx in tqdm(range(len(files)), desc = 'for all filenames', disable = TQDM_DISABLE):
        #temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL, f'cat{files[idx]}.csv'), header = None, sep= ';')#, parse_dates=[1]))
        #temp_dataframe.columns = ['target', 'timestamp', 'site']
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'user{files[idx]}.csv'), 
                                     #sep= ';', 
                                     usecols = ['site'])#, parse_dates=[1]))
        site_cnt += Counter(temp_dataframe.site.values)
        #rows += round(temp_dataframe.shape[0] / sess_len + 0.499)

    #rows = rows

    for idx, site_dict in enumerate(site_cnt.most_common(), start = 1):
        outp_freq_site_dict[site_dict[0]] = (idx, site_dict[1])    
    
    return outp_freq_site_dict


# In[16]:


get_ipython().run_cell_magic('time', '', "# 16 sec\nif not os.path.isfile(os.path.join(PATH_TO_DATA_OWN, 'site_freq_150_data.pkl')):\n    freq_site_dict = create_load_freq_site_dict(filenames_all)\n    pickle.dump(freq_site_dict, open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_150_data.pkl'), 'wb'))\n    print(len(freq_site_dict))\nelse:\n    #pkl_name = os.path.join(PATH_TO_DATA_OWN, 'site_freq_full_data.pkl')\n    freq_site_dict = pickle.load(open(os.path.join(PATH_TO_DATA_OWN, 'site_freq_150_data.pkl'), 'rb'))")


# In[ ]:





# Для начала посмотрим accuracy при использовании данных только о посещении сайтов при выбранных параметрах.    
# Подготовим данные: обучающую, тестовую и проверочную выборки.    
# Посмотрим на целевые метки.     
# Оценим точность на SGDClassifier.

# In[17]:


def prepare_sparse_train_set_window(inp_fnames, inp_sess_len, inp_wind_size, 
                                    inp_freq_site_dict, inp_time_limit = 999999999):
    
    ret_userid_list = []

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    row_index = 0
    
    for idx in tqdm(range(len(inp_fnames)), disable = TQDM_DISABLE):

        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'user{inp_fnames[idx]}.csv'), 
                                         usecols = ['site', 'time_diff']
                                        )
        userid = inp_fnames[idx]
              
        index = 0

        while index < temp_dataframe.shape[0]:
            condition_index = 0
            total_time = 0

            new_sess = defaultdict(float)
            # determine one current session
            while (condition_index < inp_sess_len) and (total_time < inp_time_limit) and ((index + condition_index) < temp_dataframe.shape[0]):
                new_sess[inp_freq_site_dict[temp_dataframe.site[index + condition_index]][0]] += 1
                total_time += temp_dataframe.time_diff[index + condition_index]
                
                condition_index += 1

            if (condition_index <= inp_wind_size):
                index += (condition_index)
                #early_exit += 1
            else:
                index += inp_wind_size

                
            row_csr += [row_index] * len(new_sess)  # row number in which value is parse
            data_csr+= list(new_sess.values())      # value
            col_csr += list(new_sess.keys())        # column number in which value is parse

            ret_userid_list.append(int(userid))
            row_index += 1

    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    #print(early_exit)
    
    return ret_csr, np.array(ret_userid_list)


# In[ ]:





# In[19]:


targ_hist = np.bincount(targets)

bin_count = []
for el in targ_hist:
    if el != 0:
        bin_count.append(el)


fig = px.box(y = bin_count)
fig.update_layout(autosize = False,
                  width = 500,
                  height= 600,
                  title = u'Статистики количества сессий для пользователей',
                  yaxis_title = u'Количество сессий',
                 )
fig.show()


# Как мы видим, есть некоторое количество пользователей для которых количество сессий аномально много.    
# При получении неудовлетворительного результата вероятно стоит произвести undesampling или oversampling.    
# На данный момент такая необходимость не очевидна. Продолжаем работу с текущими выборками.

# In[20]:


def get_metrics(inp_clf, inp_skf, inp_x_tr, inp_y_tr, inp_x_val, inp_y_val, endmodel = False):
    acc_cv  = []
    acc_val = 0
    f1_cv   = []
    f1_val = 0
    
    for train_index, test_index in tqdm(inp_skf.split(inp_x_tr, inp_y_tr), 
                                        desc = str(inp_clf).split('(')[0],
                                        disable = TQDM_DISABLE
                                       ):
        iclf = inp_clf
        iclf.fit(inp_x_tr[train_index], inp_y_tr[train_index])
        predicted = iclf.predict(inp_x_tr[test_index])
        acc_cv.append(accuracy_score(inp_y_tr[test_index], predicted))
        f1_cv.append(f1_score(inp_y_tr[test_index], predicted, average = 'macro'))
    
    if endmodel:
        iclf = inp_clf
        iclf.fit(inp_x_tr, inp_y_tr)

        predicted = iclf.predict(inp_x_val)
        acc_val = accuracy_score(inp_y_val, predicted)
        f1_val = f1_score(inp_y_val, predicted, average = 'macro')
        
        return acc_cv, f1_cv, acc_val, f1_val, iclf

    return acc_cv, f1_cv


# Произведем оценку текущей accuracy с помощью SGDClassifier

# In[21]:


get_ipython().run_cell_magic('time', '', 'clf_sgd = SGDClassifier(n_jobs = -1, random_state = rand_seed)\nsgd_accur_fold, sgd_f1_fold = get_metrics(clf_sgd, skf, X_train, y_train, X_valid, y_valid)\nsgd_accur_fold, np.mean(sgd_accur_fold), sgd_f1_fold, np.mean(sgd_f1_fold)')


# In[22]:


del train_data_sparse
del targets
del X_train
del X_valid
del y_train
del y_valid
del clf_sgd

gc.collect()


# In[ ]:





# Попытаемся ее улучшить. Для этого добавим новые признаки.     
#     
# В виду того, что необходимо производить классификацию на 150 пользователей бинарные признаки нам сильно не помогут.    
# Из неиспользуемой информации из данных у нас еще временные метки и последовательность переходов по сайтам.    
# Т.к. последовательность переходов повлечет кратное увеличение данных попробуем обойтись только данными о    
# временных метках посещения сайтов.     
# Действительно: если 2 пользователя похожи по посещенным сайтам, то, например, один может зайти проверить соцсети на    
# 1 секунду, а дальше работать с почтой, а второй, наоборот, провести много времени в соцсети, а затем на секунду    
# проверить почту. В признаках только посещенных сайтов они будут абсолютно одинаковы, но не во временных признаках.    
# 
# Добавим данные о времени нахождения пользователя на сайте (в секундах), если пользователь повторно зашел на сайт,    
# то данные временные интервалы суммируем. Минимальное время нахождения на сайте зададим равным 1 секунде. 'Нормируем'     
# эти значения на ограничение по времени, т.к. ни одно посещение не может превышать этого порога, а нормировка     
# необходима для ряда алгоритмов.    
# Возникает вопрос как поступать с последним сайтом в сессии: возможно, он на секунду вышел за пределы допустимой длины,    
# а, возможно, пользователь зашел на него на секунду и следующий посещенный сайт только через неделю. Данный момент      
# становится гиперпараметром, рассматривать который в рамках данной работы не станем.    
# 
# Так же добавим информацию о дне и часе начала сессии. Стоит обратить внимание, что из 24 часов суток сессии начинаются         
# только в 17 часах. Признаки начала сессии в остальные часы необходимо исключить, т.к. они будут нулевыми, что приводит    
# к линейной зависимости признаков, что, в свою очередь, вредит линейным методам.

# In[23]:


def prepare_sparse_train_set_total(inp_fnames, inp_sess_len, inp_wind_size, 
                                   inp_freq_site_dict, inp_time_limit = 999999999):
    
    ret_userid_list = []

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    row_index = 0

    time_start_idx = len(inp_freq_site_dict)
    
    for idx in tqdm(range(len(inp_fnames)), disable = TQDM_DISABLE):
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'user{inp_fnames[idx]}.csv'), 
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

    #check_empty_cols = set(col_csr)
    #for el in range(max(check_empty_cols)):
    #    if el not in check_empty_cols and el != 0:
    #        print(el)
            
    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    
    return ret_csr, np.array(ret_userid_list)


# Подготовим данные: обучающую, тестовую и проверочную выборки.    

# In[24]:


train_data_sparse_v2, targets_v2 = prepare_sparse_train_set_total(filenames_all, sess_len, 
                                                             wind_size, freq_site_dict,
                                                             inp_time_limit = time_limit,
                                                            )
X_train_v2, X_valid_v2, y_train_v2, y_valid_v2 = train_test_split(train_data_sparse_v2, targets_v2, 
                                                                  test_size = 0.3, random_state = rand_seed, 
                                                                  stratify = targets_v2,
                                                                 )


# Посмотрим на размерности.

# In[25]:


train_data_sparse_v2.shape


# Мы ожидаемо получили большое признаковое пространство. При этом явно не все признаки одинаково полезны.     
# Идея - выбрать лучший классификатор на текущем пространстве признаков и на его основе произвести отбор признаков.    

# Посмотрим на точности классификаторов на текущем пространстве признаков с помощью определенного ранее StratifiedKFold.

# SGDClassifier

# In[26]:


get_ipython().run_cell_magic('time', '', '# < 1 min\nclf_sgd = SGDClassifier(n_jobs = -1, random_state = rand_seed)\nsgd_v2_accur_fold, sgd_v2_f1_fold = get_metrics(clf_sgd, skf, X_train_v2, y_train_v2, X_valid_v2, y_valid_v2)\nsgd_v2_accur_fold, np.mean(sgd_v2_accur_fold), sgd_v2_f1_fold, np.mean(sgd_v2_f1_fold)')


# <i>Неожиданно низкий результат

# In[ ]:





# LogisticRegression 

# In[27]:


get_ipython().run_cell_magic('time', '', '# 17+ min\nclf_lr = LogisticRegression(n_jobs = -1, random_state = rand_seed,\n                           )\nlr_accur_fold, lr_f1_fold = get_metrics(clf_lr, skf, X_train_v2, y_train_v2, X_valid_v2, y_valid_v2)\nlr_accur_fold, np.mean(lr_accur_fold), lr_f1_fold, np.mean(lr_f1_fold)')


# <i>Крайне неожиданный низкий результат, при том, что без временных празнаков резутьтат был сопоставим с SGDClassifier

# In[ ]:





# KNeighborsClassifier

# In[28]:


get_ipython().run_cell_magic('time', '', '# 15+ min\nclf_knn = KNeighborsClassifier(n_jobs = -1)\nkn_accur_fold, kn_f1_fold = get_metrics(clf_knn, skf, X_train_v2, y_train_v2, X_valid_v2, y_valid_v2)\nkn_accur_fold, np.mean(kn_accur_fold), kn_f1_fold, np.mean(kn_f1_fold)')


# <i>Ожидаемый достойный результат.

# In[ ]:





# LinearSVC

# In[29]:


get_ipython().run_cell_magic('time', '', '# 36+ min\nclf_svc = LinearSVC(random_state = rand_seed)\nsvc_accur_fold, svc_f1_fold = get_metrics(clf_svc, skf, X_train_v2, y_train_v2, X_valid_v2, y_valid_v2)\nsvc_accur_fold, np.mean(svc_accur_fold), svc_f1_fold, np.mean(svc_f1_fold)')


# <i>Ожидаемый достойный результат.

# In[ ]:





# RandomForestClassifier

# In[30]:


get_ipython().run_cell_magic('time', '', '# 1 hour 15+ min\nclf_rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = rand_seed,)\nrf_accur_fold, rf_f1_fold, rf_accur, rf_f1, clf_rf = get_metrics(clf_rf, skf, X_train_v2, y_train_v2, X_valid_v2, y_valid_v2, endmodel = True)\nrf_accur_fold, np.mean(rf_accur_fold), rf_accur, rf_f1_fold, np.mean(rf_f1_fold), rf_f1')


# <i>Лучший результат

# In[ ]:





# XGBoost

# In[31]:


xgb_params = {
    'tree_method':'gpu_hist',
    "objective": "multi:softmax",
    "num_class": 150,
    "eval_metric": "mlogloss",
    #"verbosity": 2,
}


# XGBoost необходимы классы типа int в пределах от 0 до len(classes) - 1.     
# Подготовим.

# In[32]:


remap = {el:idx for idx, el in enumerate(set(y_train_v2))}

y_tr_v2 = [remap[x] for x in y_train_v2]
y_val_v2 = [remap[x] for x in y_valid_v2]
Dtrain = xgb.DMatrix(X_train_v2, y_tr_v2)
Dtest = xgb.DMatrix(X_valid_v2)


# In[33]:


get_ipython().run_cell_magic('time', '', "# 40+ min\nmodel_xgb = xgb.train(xgb_params, Dtrain, \n                 )\npred = model_xgb.predict(Dtest)\naccuracy_score(y_val_v2, pred), f1_score(y_val_v2, pred, average = 'macro')")


# <i>Неожидано слабый результат

# In[ ]:





# In[34]:


time.strftime('%H:%M:%S')


# Как видим лучший результат показал RandomForest.    
# Дальше работаем с ним.

# In[35]:


del clf_sgd
del clf_lr
del clf_knn
del clf_svc
del model_xgb
del y_tr_v2
del y_val_v2
del Dtrain
del Dtest

gc.collect()


# In[36]:


del X_train_v2
del X_valid_v2
del y_train_v2
del y_valid_v2

gc.collect()


# Уменьшим размерность пространства признаков.

# Для начала обучим RandomForestClassifier на всей выборке объектов для более корректного определения полезности признаков.

# In[37]:


get_ipython().run_cell_magic('time', '', 'clf_rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)\nclf_rf.fit(train_data_sparse_v2, targets_v2)')


# In[38]:


del train_data_sparse_v2
del targets_v2

gc.collect()


# Т.к. классификатор у нас уже обучен, отбор признаков будем проводить на основе их полезности через sklearn.feature_selection.SelectFromModel.

# Зададим варианты порога полезности признаков и посмотрим на изменение точности модели при отборе признаков по таким порогам.

# In[39]:


# params for select features
thres = ['mean', 1e-5, 'median', 1e-6]


# In[40]:


def check_useful(inp_list):
    
    site = 0
    sites = len(freq_site_dict)
    timings = 0
    dow = 0
    hour = 0
    for idx, el in enumerate(inp_list):
        if idx < (sites) and el == True:
            site += 1
        elif (idx >= sites) and (idx < (sites*2)) and el == True:
            timings += 1
        elif (idx >= sites*2) and (idx < sites*2 + 7) and el == True:
            dow  += 1
        elif (idx >= sites*2+7) and el == True:
            hour += 1
            
    print(u'Информативных сайтов: ' + str(site) + u', времен посещения: ' + str(timings) +          u', дней недели: ' + str(dow) + u', часов: ' + str(hour))
    
    return


# In[41]:


def prepare_sparse_train_set_total_v2(inp_fnames, inp_sess_len, inp_wind_size, 
                                   inp_freq_site_dict, inp_time_limit = 999999999):
    
    ret_userid_list = []

    row_csr  = []
    col_csr  = []
    data_csr = []
        
    row_index = 0
    #early_exit = 0
    
    time_start_idx = len(inp_freq_site_dict)
    
    for idx in tqdm(range(len(inp_fnames)), disable = TQDM_DISABLE):
        temp_dataframe = pd.read_csv(os.path.join(PATH_TO_DATA_ALL_PREP, f'user{inp_fnames[idx]}.csv'), 
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
            new_sess[inp_freq_site_dict['dow_'  + str(dow)][0]]  = 1
            new_sess[inp_freq_site_dict['hour_' + str(hour)][0]] = 1
            
            # determine one current session
            while (condition_index < inp_sess_len) and (total_time < inp_time_limit) and ((index + condition_index) < temp_dataframe.shape[0]):
                site_time = temp_dataframe.time_diff[index + condition_index]
                if site_time == 0:
                    site_time += 1
                        
                # normalize on ~[0, 1)
                new_sess[inp_freq_site_dict[temp_dataframe.site[index + condition_index]][0]] += (1/10)
                new_sess[inp_freq_site_dict['time_' + temp_dataframe.site[index + condition_index]][0]] += (site_time/inp_time_limit)
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

    #check_empty_cols = set(col_csr)
    #for el in range(max(check_empty_cols)):
    #    if el not in check_empty_cols and el != 0:
    #        print(el)
            
    ret_csr = csr_matrix((data_csr, (row_csr, col_csr)), shape = (max(row_csr)+1, max(col_csr)+1))[:, 1:]
    
    return ret_csr, np.array(ret_userid_list)


# In[42]:


def feature_selection_total(inp_feature_useful, inp_freq_site_dict):
    
    # create 'new' freq_site_dict
    ret_freq_site_low_dict = defaultdict(lambda: (0, 0.0))
    reverse_freq_site_dict = defaultdict(lambda: 'w8.it.is.incredible.page')
    for idx, el in enumerate(inp_freq_site_dict, start = 1):
        reverse_freq_site_dict[idx] = el 

    sites = len(inp_freq_site_dict)
    
    idx = 1
    for index, useful in enumerate(inp_feature_useful, start = 1):
        if useful:
            if index <= sites:
                ret_freq_site_low_dict[reverse_freq_site_dict[index]] = (idx, 0)
            elif (index > sites) and (index <= (sites*2)):
                ret_freq_site_low_dict['time_' + reverse_freq_site_dict[index - sites]] = (idx, 0)
            elif (index > sites*2) and (index <= sites*2 + 7):
                dow = index - sites*2 - 1
                #new_sess[time_start_idx*2 + dow + 1] = 1
                ret_freq_site_low_dict['dow_' + str(dow)] = (idx, 0)
            elif (index > sites*2 + WEEK):
                hour = index - sites*2 - WEEK - 1 + 7 
                #new_sess[time_start_idx*2 + WEEK + hour - 7 + 1] = 1
                ret_freq_site_low_dict['hour_' + str(hour)] = (idx, 0)
                
            idx += 1
            
    return ret_freq_site_low_dict


# In[43]:


get_ipython().run_cell_magic('time', '', "thres_acc = []\nthres_f1  = []\nfor el in thres:\n    # select features\n    features = feature_selection.SelectFromModel(clf_rf, prefit=True, threshold=el)\n    print(u'Порог информативности признаков: ' + str(el) + u', информативных признаков: ' + str(sum(features.get_support())))\n    check_useful(features.get_support())\n    \n    freq_site_low_dict = feature_selection_total(features.get_support(), freq_site_dict)\n    \n    \n    # create new train/test data\n    train_data_sparse_v3, targets_v3 = prepare_sparse_train_set_total_v2(filenames_all, sess_len, \n                                                             wind_size, freq_site_low_dict,\n                                                             inp_time_limit = time_limit,\n                                                            )\n    X_train_v3, X_valid_v3, y_train_v3, y_valid_v3 = train_test_split(train_data_sparse_v3, targets_v3, \n                                                                      test_size = 0.3, \n                                                                      random_state = rand_seed, stratify = targets_v3,\n                                                                     )\n    \n    # evaluate results\n    clf_rf_v3 = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = rand_seed)\n    rf_v3_accur_fold, rf_v3_f1_fold = get_metrics(clf_rf_v3, skf, X_train_v3, y_train_v3, X_valid_v3, y_valid_v3)\n    thres_acc.append(np.mean(rf_v3_accur_fold))\n    thres_f1.append(np.mean(rf_v3_f1_fold))\n    \n    print(rf_v3_accur_fold, np.mean(rf_v3_accur_fold), '\\n', rf_v3_f1_fold, np.mean(rf_v3_f1_fold), '\\n\\n')")


# In[44]:


del train_data_sparse_v3
del targets_v3
del X_train_v3
del X_valid_v3
del y_train_v3
del y_valid_v3
del clf_rf_v3

gc.collect()

 было
 [0.8311889770964505, 0.8289507703070582, 0.8280534728453455],
 0.8293977400829514,
 0.8749112845990064,
 [0.8183393049812279, 0.8147882586838809, 0.8147106707020927],
 0.8159460781224005,
 0.866244659289942
 
 получили при 1e-06
 [0.8285275420506737, 0.8280230559822366, 0.8267759645947713] 
 0.8277755208758939 
 [0.8146833289269194, 0.8138646569729491, 0.8135717575544326] 
 0.8140399144847671
 
 получили при median
 [0.8280712960428263, 0.8282663908871079, 0.8253615804602071] 
 0.827233089130047 
 [0.8144491604206092, 0.8135619777183454, 0.811205626551376] 
 0.8130722548967769 
# In[60]:


1e-6, 0.8293977400829514 - 0.8277755208758939, 0.8159460781224005 - 0.8140399144847671


# In[63]:


'median', 0.8293977400829514 - 0.827233089130047  , 0.8159460781224005 - 0.8130722548967769


# Сократив количество признаков с 55618 до 34142 мы потеряли в средней accuracy на  CV 0.00162 и на f1-score 0.00190.  
# Дальнейшее сокращение до 27809 признаков привело к потере в средней accuracy на CV 0.00216 и на f1-score 0.00287.

# In[47]:


time.strftime('%H:%M:%S')


# In[ ]:





# ## Попробуем улучшить.  
# Выберем порог 1е-6.  
# Перебор по сетке параметров количества листьев, глубины, признаков и т.д. возможен, но займет слишком много времени.  
# Вспомним, что RandomForest не переобучаются при увеличении числа деревьев. Воспользуемся этим моментом и только  
# увеличим количество деревьев до 200.

# In[49]:


get_ipython().run_cell_magic('time', '', "features = feature_selection.SelectFromModel(clf_rf, prefit = True, threshold = 1e-6)\nprint('feature importance threshold: ' + str(el) + ', useful features: ' + str(sum(features.get_support())))\ncheck_useful(features.get_support())\n\nfreq_site_low_dict = feature_selection_total(features.get_support(), freq_site_dict)\n\ntrain_data_sparse_v4, targets_v4 = prepare_sparse_train_set_total_v2(filenames_all, sess_len, \n                                                         wind_size, freq_site_low_dict,\n                                                         inp_time_limit = time_limit,\n                                                        )\nX_train_v4, X_valid_v4, y_train_v4, y_valid_v4 = train_test_split(train_data_sparse_v4, targets_v4, \n                                                                  test_size = 0.3, \n                                                                  random_state = rand_seed, stratify = targets_v4,\n                                                                 )")


# In[50]:


get_ipython().run_cell_magic('time', '', '#n_estimators=100,\nclf_rf_200 = RandomForestClassifier(n_estimators = 200, n_jobs = -1, random_state = rand_seed)\nrf200_accur_fold, rf200_f1_fold, rf200_accur, rf200_f1, clf_rf_200 = \\\n                                      get_metrics(clf_rf_200, skf, X_train_v4, y_train_v4, \n                                                  X_valid_v4, y_valid_v4, endmodel = True,\n                                                  )\nrf200_accur_fold, np.mean(rf200_accur_fold), rf200_accur, rf200_f1_fold, np.mean(rf200_f1_fold), rf200_f1')


# In[ ]:





# In[53]:


time.strftime('%H:%M:%S')


# ### Выводы:    
# - удалось реализовать идентификацию пользователя из заданных 150 по последовательности посещенных им сайтов в течении времени,     
#   не превышающем 10 минут;
# - достигнутые значения метрик:  
#   - accuracy 0.8330 на CV и 0.8794 на валидационной; 
#   - f1_score(macro) 0.8195 на CV и 0.8714 на валидационной.

# ### Возможные пути улучшения:
# - добавление дополнительных признаков с учетом порядка посещенных сайтов (n-граммы) с последующим их отбором по полезности;
# - обучение 150 различных классификаторов для каждого пользователя в отдельности (one-vs-rest) со своим признаковым пространством;
# - дополнительная оптимизация гиперпараметров выбранного алгоритма;
# - использование ансамбля различных алгоритмов;
# - работа с балансами классов;
# - получение дополнительных данных по поведению пользователя (геолокация, устройства и т.п.);

# In[ ]:




