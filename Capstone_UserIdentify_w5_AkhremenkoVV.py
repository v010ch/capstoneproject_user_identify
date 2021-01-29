#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/web/677/8e1/337/6778e1337c3d4b159d7e99df94227cb2.jpg"/>
# ## Специализация "Машинное обучение и анализ данных"
# <center>Автор материала: программист-исследователь Mail.Ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ [Юрий Кашницкий](https://yorko.github.io/)

# # <center> Capstone проект №1 <br> Идентификация пользователей по посещенным веб-страницам
# <img src='http://i.istockimg.com/file_thumbview_approve/21546327/5/stock-illustration-21546327-identification-de-l-utilisateur.jpg'>
# 
# # <center>Неделя 5.  Соревнование Kaggle "Catch Me If You Can"
# 
# На этой неделе мы вспомним про концепцию стохастического градиентного спуска и опробуем классификатор Scikit-learn SGDClassifier, который работает намного быстрее на больших выборках, чем алгоритмы, которые мы тестировали на 4 неделе. Также мы познакомимся с данными [соревнования](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) Kaggle по идентификации пользователей и сделаем в нем первые посылки. По итогам этой недели дополнительные баллы получат те, кто попадет в топ-30 публичного лидерборда соревнования.

# In[70]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[71]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,pandas,matplotlib,statsmodels,sklearn -g')


# In[ ]:





# In[3]:


import os
import pandas as pd
import pickle
from collections import Counter
import numpy as np

from tqdm import tqdm
tqdm.pandas()

from scipy.sparse import csr_matrix
from scipy import sparse


# In[81]:


from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC


import catboost
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# In[5]:


import plotly.express as px


# In[6]:


PATH_TO_DATA = os.path.join('./', 'data', 'compete-catch_me_if_you_can')
PATH_TO_DATA


# In[7]:


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# In[ ]:





# **Считаем данные [соревнования](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2) в DataFrame train_df и test_df (обучающая и тестовая выборки).**

# In[8]:


train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                       index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
                      index_col='session_id')

train_df.shape, test_df.shape


# In[9]:


train_df.head()


# In[10]:


train_df.keys()


# In[11]:


train_test_df = pd.concat([train_df, test_df])


# **Пока для прогноза будем использовать только индексы посещенных сайтов. Индексы нумеровались с 1, так что заменим пропуски на нули.**

# In[12]:


train_test_df_sites = train_test_df[['site%d' % i for i in range(1, 11)]].fillna(0).astype('int')


# In[13]:


train_test_df_sites.shape


# ***Создаем разреженную матрицу***

# In[14]:


train_test_df_sites.reset_index(inplace=True)


# In[15]:


get_ipython().run_cell_magic('time', '', 'row_csr  = []\ndata_csr = []\ncol_csr  = []\n\nfor idx in range(train_test_df_sites.shape[0]):\n    for el in Counter(train_test_df_sites.loc[idx, :].values[1:]).most_common():\n        row_csr.append(idx)\n        data_csr.append(el[1])\n        col_csr.append(el[0])\n    ')


# In[16]:


train_test_sparse = csr_matrix((data_csr, (row_csr, col_csr)), shape=(max(row_csr)+1, max(col_csr)+1))


# In[17]:


train_test_sparse.shape


# Выделяем отдельные вектора. Игнорируем колонку с сайтом 0

# In[18]:


X_train_sparse = train_test_sparse[:train_df.shape[0], 1:]
X_test_sparse = train_test_sparse[train_df.shape[0] :, 1:]
y = train_df.target


# In[19]:


print('was' + str(train_df.shape) + str(test_df.shape))
print(X_train_sparse.shape, X_test_sparse.shape)


# **<font color='red'>Вопрос 1. </font> Выведите размерности матриц *X_train_sparse* и *X_test_sparse* – 4 числа на одной строке через пробел: число строк и столбцов матрицы *X_train_sparse*, затем число строк и столбцов матрицы *X_test_sparse*.**

# In[20]:


print(X_train_sparse.shape, X_test_sparse.shape)


# In[21]:


with open(os.path.join(PATH_TO_DATA, 'X_train_sparse.pkl'), 'wb') as X_train_sparse_pkl:
    pickle.dump(X_train_sparse, X_train_sparse_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'X_test_sparse.pkl'), 'wb') as X_test_sparse_pkl:
    pickle.dump(X_test_sparse, X_test_sparse_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'train_target.pkl'), 'wb') as train_target_pkl:
    pickle.dump(y, train_target_pkl, protocol=2)


# In[22]:


y.shape


# Разобьем выборки с учетом времение, не перемешивая

# In[23]:


train_share = int(.7 * X_train_sparse.shape[0])
X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]
X_valid, y_valid  = X_train_sparse[train_share:, :], y[train_share:]


# In[24]:


X_train.shape, y_train.shape


# In[25]:


type(X_train), type(y_train)


# In[26]:


y.head()


# ***Пробуем обучить SGDClassifier. В начале на всей выборке и оценим точность. Затем по всей выборке и сделаем  прогноз для соревнования***

# In[27]:


sgd_logit = SGDClassifier(loss = 'log', random_state = 17, n_jobs = -1)
sgd_logit.fit(X_train, y_train)


# In[28]:


logit_valid_pred_proba = sgd_logit.predict_proba(X_valid)


# In[29]:


logit_valid_pred_proba[:,1]


# **<font color='red'>Вопрос 2. </font> Посчитайте ROC AUC логистической регрессии, обученной с помощью стохастического градиентного спуска, на отложенной выборке. Округлите до 3 знаков после разделителя.**

# In[91]:


roc_auc = roc_auc_score(y_valid, logit_valid_pred_proba[:,1])
print(round(roc_auc, 3)) #, sum(predicted), sum(y_train))


# In[ ]:





# In[33]:


sgd_logit_full = SGDClassifier(loss = 'log', random_state = 17, n_jobs = -1)
sgd_logit_full.fit(X_train_sparse, y)


# In[34]:


logit_valid_pred_proba_full = sgd_logit_full.predict_proba(X_test_sparse)


# In[ ]:





# Сохраним и отправим на соревновние. Посмотрим результат. 'Команда' **[YDF & MIPT] Coursera_Akhremenko Vladimir**

# In[35]:


write_to_submission_file(logit_valid_pred_proba_full[:, 1], 'submission_avv.csv')


# текущий скор: ***0.91646***    
# Первый требуемый бенчмарк уже превзойден: **0.91273 sgd_logit_benchmark.csv** 

# Доступные бенчмарки
# 
# 0.95965    A3 strong baseline (20 credits)    
# 0.95343    A3 baseline (10 credits)    
# 0.95216    Logit Tf-Idf 6 features    
# **0.92784    Logit +3 features**    
# 0.92784    A3 baseline 2    
# 0.92692    CountVectorizer-logit-3feat    
# **0.91273    sgd_logit_benchmark.csv**    
# 0.91252    Alice - logistic regression baseline    
# 0.90812    A3 baseline 1    
# 0.90703    CountVectorizer-logit    
# 

# In[ ]:





# In[ ]:





# In[ ]:





# # Попробуем улучшить результат и превзойти второй требуемый бенчмарк: 0.92784 Logit +3 features    
# Создадим собственные признаки

# In[36]:


with open(os.path.join(PATH_TO_DATA, 'X_train_sparse.pkl'), 'rb')as f:
    X_train_sparse = pickle.load(f)
    
with open(os.path.join(PATH_TO_DATA, 'X_test_sparse.pkl'), 'rb')as f:
    X_test_sparse = pickle.load(f)


# In[37]:


train_df[['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
          'site7', 'site8', 'site9', 'site10']] = \
        train_df[['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
                  'site7', 'site8', 'site9', 'site10']].fillna(0).astype('int')

train_df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 
          'time7', 'time8', 'time9', 'time10']] =\
        train_df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 
                  'time7', 'time8', 'time9', 'time10']].fillna(0)

#train_df.head()


# In[38]:


test_df[['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
          'site7', 'site8', 'site9', 'site10']] = \
        test_df[['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
                 'site7', 'site8', 'site9', 'site10']].fillna(0).astype('int')

test_df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 
          'time7', 'time8', 'time9', 'time10']] = \
        test_df[['time1', 'time2', 'time3', 'time4', 'time5', 'time6', 
                 'time7', 'time8', 'time9', 'time10']].fillna(0)

#test_df.head()


# In[39]:


site_ftrs = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6', 
          'site7', 'site8', 'site9', 'site10']
time_ftrs = ['time1', 'time2','time3','time4','time5','time6',
             'time7','time8','time9','time10']
zero_dt = pd.to_datetime(0)


# In[40]:


for el in time_ftrs:
    train_df[el] = pd.to_datetime(train_df[el])
    test_df[el] = pd.to_datetime(test_df[el])
    


# Соберем статистики по рабочим: дням и выходным по отдельности:       
# - [x] количество уникальных сайтов в сессии
# - [x] длительность посещения первого сайта в сессии
# - [x] длительность сессии
# - [x] час начала сессии
# - [x] день недели начала сессии
# - [ ] длительность посещения первого сайта в день mean/std
# - [ ] время входа на первый сайт в день mean/std
# 
# (галочкой отмечены фичи, участвующие в финальном решении)

# In[41]:


def get_stats(inp_data, new_df, target = None):
    
    day_of_week = []
    session_timespan = []
    start_hour = []
    unique_sites = []

    # statistics per user at weekdays
    #first_day_site  = []         # first site of the day
    #first_sess_site = []         # first site in session
    weekdays_first_day_site_duration = []   # duration on first site of the day
    weekdays_first_sess_site_duration = []  # duration on first site in session
    weekdays_first_connect_in_day_time = [] # time of enter at the first site of the day
    weekdays_start_hour = []
    weekdays_unique_sites = []
    weekdays_session_timespan = []

    # statistics per user at weekends
    #first_day_site  = []         # first site of the day
    #first_sess_site = []         # first site in session
    weekend_first_day_site_duration = []   # duration on first site of the day
    weekend_first_sess_site_duration = []  # duration on first site in session
    weekend_first_connect_in_day_time = [] # time of enter at the first site of the day
    weekend_start_hour = []
    weekend_unique_sites = []
    weekend_session_timespan = []
    
    if isinstance(target, np.int):
        indexes = inp_data[inp_data.target == target].index
    else:
        indexes = inp_data.index
    
    for el in tqdm(indexes):
       
        #!!!!!!!!!! site10 == 0
        #if 0 in inp_data.loc[el, site_ftrs].unique():
        if inp_data.loc[el, 'site10'] == 0:
             uniq_sites = len(inp_data.loc[el, site_ftrs].unique()) - 1
        else:
            uniq_sites = len(inp_data.loc[el, site_ftrs].unique())
        new_df.loc[el, 'unique_sites'] = uniq_sites

        if inp_data.loc[el, 'time2'] != zero_dt:
            site_duration = (inp_data.loc[el, 'time2'] - inp_data.loc[el, 'time1']).total_seconds()
        else:
            site_duration = 0
        new_df.loc[el, 'first_sess_site_duration'] = site_duration

        for time_ftr in list(time_ftrs[::-1]):
            if inp_data.loc[el, time_ftr] != zero_dt:
                session_timespan = (inp_data.loc[el, time_ftr] - inp_data.loc[el, 'time1']).total_seconds()
                break
        new_df.loc[el, 'session_timespan'] = session_timespan 

        #if (train_df.loc[el, 'time1'].weekday() == 5) or ((train_df.loc[el, 'time1'].weekday() == 6)):
        if (new_df.loc[el, 'dow'] == 5) or ((new_df.loc[el, 'dow'] == 6)):
            weekend_unique_sites.append(uniq_sites)   
            weekend_first_sess_site_duration.append(site_duration)
            weekend_start_hour.append(new_df.loc[el, 'start_hour'])
            weekend_session_timespan.append(session_timespan)
        else:
            weekdays_unique_sites.append(uniq_sites)
            weekdays_first_sess_site_duration.append(site_duration)
            weekdays_start_hour.append( new_df.loc[el, 'start_hour'])
            weekdays_session_timespan.append( session_timespan )

    stat_dict = {'weekend_unique_sites_mean' : np.mean(weekend_unique_sites)
                         ,'weekend_unique_sites_std' : np.std(weekend_unique_sites)
                         ,'weekend_first_sess_site_duration_mean' : np.mean(weekend_first_sess_site_duration)
                         ,'weekend_first_sess_site_duration_std' : np.std(weekend_first_sess_site_duration)
                         ,'weekend_start_hour_mean' : np.mean(weekend_start_hour)
                         ,'weekend_start_hour_std' : np.std(weekend_start_hour)
                         ,'weekend_session_timespan_mean' : np.mean(weekend_session_timespan)
                         ,'weekend_session_timespan_std' : np.std(weekend_session_timespan)

                         ,'weekdays_unique_sites_mean' : np.mean(weekdays_unique_sites)
                         ,'weekdays_unique_sites_std' : np.std(weekdays_unique_sites)
                         ,'weekdays_first_sess_site_duration_mean' : np.mean(weekdays_first_sess_site_duration)
                         ,'weekdays_first_sess_site_duration_std' : np.std(weekdays_first_sess_site_duration)
                         ,'weekdays_start_hour_mean' : np.mean(weekdays_start_hour)
                         ,'weekdays_start_hour_std' : np.std(weekdays_start_hour)
                         ,'weekdays_session_timespan_mean' : np.mean(weekdays_session_timespan)
                         ,'weekdays_session_timespan_std' : np.std(weekdays_session_timespan)    
                         }
    return new_df, stat_dict


# In[42]:


def get_stats_sparse(inp_data, sparse_data):
    
    tmp_df = pd.DataFrame(index = inp_data.index)
    tmp_df['start_hour'] = inp_data['time1'].map(lambda x: x.hour)
    tmp_df['dow'] = inp_data['time1'].map(lambda x: x.weekday())
    
    tmp_df, stat_dict_0 = get_stats(inp_data, tmp_df, 0)
    tmp_df, stat_dict_1 = get_stats(inp_data, tmp_df, 1)
    
    return sparse_data, tmp_df, stat_dict_0, stat_dict_1


# In[43]:


get_ipython().run_cell_magic('time', '', '# 16min 26s\nX_train_sparse, train_df_add, other_stats, Alice_stats = get_stats_sparse(train_df, X_train_sparse)')


# In[44]:


get_ipython().run_cell_magic('time', '', "for el in time_ftrs:\n    test_df[el] = pd.to_datetime(test_df[el])\n\ntst_df = pd.DataFrame(index = test_df.index)\ntst_df['start_hour'] = test_df['time1'].map(lambda x: x.hour)\ntst_df['dow'] = test_df['time1'].map(lambda x: x.weekday())\n    \ntest_df_add, _ = get_stats(test_df, tst_df)")


# In[ ]:


#Alice_stats


# In[46]:


with open(os.path.join(PATH_TO_DATA, 'alice_stats.pkl'), 'wb') as af:
    pickle.dump(Alice_stats, af, protocol=2)
    
with open(os.path.join(PATH_TO_DATA, 'other_stats.pkl'), 'wb') as of:
    pickle.dump(other_stats, of, protocol=2)
    
with open(os.path.join(PATH_TO_DATA, 'X_train.pkl'), 'wb') as xf:
    pickle.dump(X_train, xf, protocol=2)


# Преобразуем категориальные фичи

# In[47]:


lb_dow = LabelBinarizer().fit(train_df_add.dow)
lb_start_hour  = LabelBinarizer().fit(train_df_add.start_hour)
lb_unique_site = LabelBinarizer().fit(train_df_add.unique_sites)


# Объеденим собранные данные в итоговые матрицы

# In[48]:


X_train_sparse_v2 = sparse.hstack((X_train_sparse, np.array(train_df_add.first_sess_site_duration)[:,None]))
X_test_sparse_v2  = sparse.hstack((X_test_sparse,  np.array(test_df_add.first_sess_site_duration)[:,None]))

X_train_sparse_v2 = sparse.hstack((X_train_sparse_v2, np.array(train_df_add.session_timespan.values)[:,None]))
X_test_sparse_v2  = sparse.hstack((X_test_sparse_v2,  np.array(test_df_add.session_timespan.values)[:,None]))

X_train_sparse_v2 = sparse.hstack((X_train_sparse_v2, 
                                   csr_matrix(lb_dow.transform(train_df_add.dow.values))))
X_test_sparse_v2  = sparse.hstack((X_test_sparse_v2,  
                                   csr_matrix(lb_dow.transform(test_df_add.dow.values))))

X_train_sparse_v2 = sparse.hstack((X_train_sparse_v2, 
                                   csr_matrix(lb_start_hour.transform(train_df_add.start_hour.values))))
X_test_sparse_v2  = sparse.hstack((X_test_sparse_v2,  
                                   csr_matrix(lb_start_hour.transform(test_df_add.start_hour.values))))

X_train_sparse_v2 = sparse.hstack((X_train_sparse_v2, 
                                   csr_matrix(lb_unique_site.transform(train_df_add.unique_sites.values))))
X_test_sparse_v2  = sparse.hstack((X_test_sparse_v2,  
                                   csr_matrix(lb_unique_site.transform(test_df_add.unique_sites.values))))


# In[49]:


get_ipython().run_cell_magic('time', '', "with open(os.path.join(PATH_TO_DATA, 'X_train.pkl'), 'rb') as xf:\n    X_train = pickle.load(xf)\n    \nwith open(os.path.join(PATH_TO_DATA, 'train_target.pkl'), 'rb') as train_target_pkl:\n    y = pickle.load(train_target_pkl)")


# In[50]:


get_ipython().run_cell_magic('time', '', 'X_tr, X_val, y_tr, y_val = train_test_split(\n    X_train_sparse_v2, y, test_size=0.33, random_state=944)')


# Так же попробуем оценить количество сессий Alice в LB на Kaggle.    
# Так как у нас других данных нет, за оценку возьмем пропорцию в тестовой выборке:    

# In[76]:


proportion = sum(y)/X_train_sparse_v2.shape[0]
proportion


# In[77]:


str(int(X_test_sparse_v2.shape[0] * proportion)) + ' from ' + str(X_test_sparse_v2.shape[0])


# ***750*** - это для нас удинственный ориентир, который мы имеем

# ***Обучим SGDClassifier на расширенных данных***

# In[51]:


clf_sgd_part = SGDClassifier(loss = 'log', random_state = 17, n_jobs = -1)
clf_sgd_part.fit(X_tr, y_tr)


# In[52]:


clf_sgd_valid_pred_proba = clf_sgd_part.predict_proba(X_val)
roc_auc_score(y_val, clf_sgd_valid_pred_proba[:,1])


# Как мы видим, результат даже немного ухудшился. В основном из-за того, что мы отошли от сортироваки по времени    
# и разбили данные на train и test через train_test_split.    
# Более того, даже изменяя random_state на отличный от 17 мы понижаем результат.    
# На Kaggele результат так же понижается, что выглядит несколько странным.

# Сделаем предположение, что это в силу стохастической природы SGDClassifier.
# Попробуем заменить его на другие классификаторы.

# ***Будм пробовать классификаторы: SGDClassifier, LogisticRegression, RandomForestClassifier, CatBoostClassifier, LighGBM.    
# Так же попробуем повысить вес класса  Alice***

# In[149]:


clf_sgd = SGDClassifier(loss = 'log', random_state = 17, n_jobs = -1, class_weight={0:0.4, 1: 0.6})
clf_sgd.fit(X_train_sparse_v2, y)

predicted_sgd = clf_sgd.predict_proba(X_train_sparse_v2)
roc_auc_sgd = roc_auc_score(y, predicted_sgd[:,1])

pred_sgd = clf_sgd.predict(X_train_sparse_v2)
pred_sgd_subm = clf_sgd.predict(X_test_sparse_v2)

print(roc_auc_sgd, sum(pred_sgd), sum(y), sum(pred_sgd_subm))

class_weight='None'
0.8909284115324846 1396 2297 402

class_weight='balanced'
0.9356726054366468 22379 2297 8605

class_weight={0:0.4, 1: 0.6}
0.9073209180461081 1278 2297 362
# In[115]:


clf_sgd_pred_proba = clf_sgd.predict_proba(X_test_sparse)


# In[116]:


write_to_submission_file(clf_sgd_pred_proba[:, 1], 'subm_avv_sgd_5ftrs2.csv')


# In[ ]:





# In[147]:


get_ipython().run_cell_magic('time', '', 'clfLR = LogisticRegression(random_state=17, n_jobs=-1, class_weight={0:0.4, 1: 0.6})\nclfLR.fit(X_train_sparse_v2, y)\n\npred_LR = clfLR.predict_proba(X_train_sparse_v2)\nroc_auc_LR = roc_auc_score(y, pred_LR[:,1])\n\npred_LR = clfLR.predict(X_train_sparse_v2)\npred_LR_test = clfLR.predict(X_test_sparse_v2)\n\nprint(roc_auc_LR, sum(pred_LR), sum(y), sum(pred_LR_test))')

class_weight = None
0.9928646995011767 1310 2297 285

class_weight={0:0.4, 1: 0.6}
0.9923592186429573 1525 2297 347
# In[150]:


predicted_LR = clfLR.predict_proba(X_test_sparse_v2)
write_to_submission_file(predicted_LR[:, 1], 'subm_avv_lr_5ftrs_w_4_6.csv')


# In[ ]:





# In[154]:


get_ipython().run_cell_magic('time', '', '# 3min 3s\n#  1min 27s\n\n#clfRF = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, random_state=17)\nclfRF = RandomForestClassifier(n_estimators=5, n_jobs=-1, random_state=17, class_weight={0:0.4, 1: 0.6})\nclfRF.fit(X_train_sparse_v2, y)\n\npred_RF = clfRF.predict_proba(X_train_sparse_v2)\nroc_auc_RF = roc_auc_score(y, pred_RF[:,1])\n\npred_RF = clfRF.predict(X_train_sparse_v2)\npred_RF_test = clfRF.predict(X_test_sparse_v2)\n\nprint(roc_auc_RF, sum(pred_RF), sum(y), sum(pred_RF_test))')

class_weight = None, est = 100
0.999999991336792 2293 2297 45

class_weight = None, est = 50
0.9999999896041504 2292 2297 52

class_weight={0:0.4, 1: 0.6}, est = 50
0.9999999792083009 2289 2297 65

class_weight={0:0.4, 1: 0.6}, est = 25
0.9999999514860355 2270 2297 69
# In[151]:


predicted_RF = clfRF.predict_proba(X_test_sparse_v2)
write_to_submission_file(predicted_RF[:, 1], 'subm_avv_rf_5ftrs_w_4_6.csv')


# In[ ]:





# In[90]:


get_ipython().run_cell_magic('time', '', '\nclf_lgbm = LGBMClassifier(class_weight={0:0.4, 1: 0.6}, n_estimators=25)\nclf_lgbm.fit(X_train_sparse_v2, y)\n\npred_lgbm = clf_lgbm.predict_proba(X_train_sparse_v2)\nroc_auc_lgbm = roc_auc_score(y, pred_lgbm[:,1])\n\npred_lgbm = clf_lgbm.predict(X_train_sparse_v2)\npred_lgbm_test = clf_lgbm.predict(X_test_sparse_v2)\n\nprint(roc_auc_lgbm, sum(pred_lgbm), sum(y), sum(pred_lgbm_test))')

class_weight = None
0.99820548664247 1702 2297 313

class_weight={0:0.4, 1: 0.6}, est = 100
0.9975441191538457 1847 2297 313

class_weight={0:0.4, 1: 0.6}, est = 25
0.9916571739276641 1557 2297 295

class_weight={0:0.4, 1: 0.6}, est = 5
0.9656990070134006 1200 2297 250
# In[88]:


pred_lgbm_test = clf_lgbm.predict(X_test_sparse_v2)
write_to_submission_file(pred_lgbm_test, 'subm_avv_lgbm_5ftrs_w_4_6.csv')


# In[ ]:





# In[53]:


ctb_clf = CatBoostClassifier(loss_function='Logloss', class_weights=[0.4, 0.6])
ctb_clf.fit(X_train_sparse_v2, y)


# In[54]:


pred_cb = ctb_clf.predict_proba(X_train_sparse_v2)
roc_auc_cb = roc_auc_score(y, pred_cb[:, 1])

pred_cb = ctb_clf.predict(X_train_sparse_v2)
pred_cb_test = ctb_clf.predict(X_test_sparse_v2)

print(roc_auc_cb, sum(pred_cb), sum(y), sum(pred_cb_test))

CatBoostClassifier(loss_function='Logloss', class_weights=[0.4, 0.6])
0.9965801137572076 1986 2297 249

CatBoostClassifier(loss_function='Logloss', class_weights=[0.35, 0.65])
0.9973392221570319 2106 2297 291

CatBoostClassifier(loss_function='Logloss', class_weights=[0.3, 0.7])
0.9980240045641384 2228 2297 314

# In[ ]:





# In[160]:


ctb_pred = ctb_clf.predict_proba(X_test_sparse_v2)


# In[161]:


write_to_submission_file(ctb_pred[:, 1], 'subm_avv_ctb_5ftrs_w_4_6.csv')


# In[ ]:





# Как мы видим, все методы, основанные на деревьях очень быстро находят критически важные признаки и в моменте переобучаются даже при малом количестве n_estimtors. Даже без кросс валидации видно, что в прогнозе они дают малое количество сессии (по отношению к нашему прогнозу в ~750) для Alice.    

# Так же сильное изменение весов класса приводило к тому, что алгоритмы выдавали сильно большое число сессий для Alice - порядка десятка тысяч из 82000+. Остановился на class_weight={0:0.4, 1: 0.6}.

# Лучший результат из опробованных алгоритмов дал LogisticRegression с измененными весами class_weight={0:0.4, 1: 0.6}.
# Как мы знаем, его можно еще улучшить через оптимизацию параметра С. Мы не будем этого делать, т.к. полученный результат ***0.94474*** позволяет опередить второй требуемый бенчмарк ***0.92784 Logit +3 features***

# **[лидерборд](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/leaderboard/public).** (текущее место 2220)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


for el in time_ftrs:
    train_df[el] = pd.to_datetime(train_df[el])
    test_df[el] = pd.to_datetime(test_df[el])


# In[11]:


train_df['start_hour'] = train_df['time1'].map(lambda x: x.hour)
train_df['dow'] = train_df['time1'].map(lambda x: x.weekday())


# In[25]:


px.bar(x = train_df[train_df.target == 1].dow.value_counts().keys()
      , y = train_df[train_df.target == 1].dow.value_counts()
      )


# In[26]:


px.bar(x = train_df[train_df.target == 0].dow.value_counts().keys()
      , y = train_df[train_df.target == 0].dow.value_counts()
      )


# In[27]:


px.bar(x = train_df[train_df.target == 1].start_hour.value_counts().keys()
      , y = train_df[train_df.target == 1].start_hour.value_counts()
      )


# In[28]:


px.bar(x = train_df[train_df.target == 0].start_hour.value_counts().keys()
      , y = train_df[train_df.target == 0].start_hour.value_counts()
      )


# In[37]:


train_df_add.keys()


# In[39]:


px.bar(x = train_df_add[train_df.target == 0].unique_sites.value_counts().keys()
      , y = train_df_add[train_df.target == 0].unique_sites.value_counts()
      )


# In[40]:


px.bar(x = train_df_add[train_df.target == 1].unique_sites.value_counts().keys()
      , y = train_df_add[train_df.target == 1].unique_sites.value_counts()
      )


# In[42]:


px.bar(x = train_df_add[train_df.target == 1].session_timespan.value_counts().keys()
      , y = train_df_add[train_df.target == 1].session_timespan.value_counts()
      )


# In[43]:


px.bar(x = train_df_add[train_df.target == 0].session_timespan.value_counts().keys()
      , y = train_df_add[train_df.target == 0].session_timespan.value_counts()
      )


# In[44]:


px.bar(x = train_df_add[train_df.target == 1].first_sess_site_duration.value_counts().keys()
      , y = train_df_add[train_df.target == 1].first_sess_site_duration.value_counts()
      )


# In[45]:


px.bar(x = train_df_add[train_df.target == 0].first_sess_site_duration.value_counts().keys()
      , y = train_df_add[train_df.target == 0].first_sess_site_duration.value_counts()
      )


# In[ ]:




