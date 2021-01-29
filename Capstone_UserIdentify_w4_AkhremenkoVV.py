#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[2]:


get_ipython().run_line_magic('watermark', '-v -m -p numpy,scipy,pandas,matplotlib,statsmodels,sklearn -g')


# In[3]:


import os
import pickle
import pandas as pd
import numpy as np
import itertools
import warnings
#from tqdm.notebook import tqdm
from tqdm import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from sklearn.model_selection import learning_curve


# In[4]:


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score


# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC


# In[ ]:





# In[6]:


PATH_TO_DATA = './my_data'
ANSW = './answers'


# In[7]:


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


# In[ ]:





# ## Часть 1. Сравнение нескольких алгоритмов на сессиях из 10 сайтов

# In[8]:


with open(os.path.join(PATH_TO_DATA, 
         'X_sp_10users.pkl'), 'rb') as X_sparse_10users_pkl:
    X_sparse_10users = pickle.load(X_sparse_10users_pkl)
with open(os.path.join(PATH_TO_DATA, 
                       'y_10users.pkl'), 'rb') as y_10users_pkl:
    y_10users = pickle.load(y_10users_pkl)

X_sparse_10users.shape


# In[11]:


X_train, X_valid, y_train, y_valid = train_test_split(X_sparse_10users, y_10users, 
                                                      test_size=0.3, 
                                                     random_state=17, stratify=y_10users)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)


# In[12]:


def plot_validation_curves(param_values, grid_cv_results_):
    train_mu, train_std = grid_cv_results_['mean_train_score'], grid_cv_results_['std_train_score']
    #valid_mu, valid_std = grid_cv_results_['mean_test_score'], grid_cv_results_['std_test_score']
    train_line = plt.plot(param_values, train_mu, '-', label='train', color='green')
    #valid_line = plt.plot(param_values, valid_mu, '-', label='test', color='red')
    plt.fill_between(param_values, train_mu - train_std, train_mu + train_std, edgecolor='none',
                     facecolor=train_line[0].get_color(), alpha=0.2)
    #plt.fill_between(param_values, valid_mu - valid_std, valid_mu + valid_std, edgecolor='none',
    #                 facecolor=valid_line[0].get_color(), alpha=0.2)
    plt.legend()


# In[ ]:





# KNeighborsClassifier

# In[13]:


def get_metrics(inp_clf, inp_skf, inp_x_tr, inp_y_tr, inp_x_val, inp_y_val):
    acc_cv = []
    acc_val = 0
    
    for train_index, test_index in skf.split(inp_x_tr, inp_y_tr):
        clf = inp_clf
        clf.fit(inp_x_tr[train_index], inp_y_tr[train_index])
        predicted = clf.predict(inp_x_tr[test_index])
        acc_cv.append(accuracy_score(inp_y_tr[test_index], predicted))
        #print('Score: ', clf.score(inp_x_tr[test_index], inp_y_tr[test_index]))
    
    clf = inp_clf
    clf.fit(inp_x_tr, inp_y_tr)

    predicted = clf.predict(inp_x_val)
    acc_val = accuracy_score(inp_y_val, predicted)

    return acc_cv, acc_val


# In[14]:


answ1_1, answ1_2 = get_metrics(KNeighborsClassifier(n_neighbors=100, n_jobs=-1), skf, X_train, y_train, X_valid, y_valid)


# In[15]:


answ1 = [round(np.mean(answ1_1), 2), round(answ1_2, 2)]
answ1


# In[ ]:





# RandomForestClassifier

# In[16]:


clfRF = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, random_state=17)


# In[17]:


oob_score, acc_rf = get_metrics(clfRF, skf, X_train, y_train, X_valid, y_valid)
oob_score, acc_rf


# In[18]:


answ2 = [round(np.mean(oob_score), 2), round(acc_rf, 2)]
answ2


# In[19]:


write_answer_to_file(answ2, 'answer4_2.txt')


# In[ ]:





# LogisticRegression

# In[20]:


clfLR = LogisticRegression(random_state=17, n_jobs=-1)


# In[21]:


acc_cv_lr, acc_lr = get_metrics(clfLR, skf, X_train, y_train, X_valid, y_valid)
acc_cv_lr, acc_lr


# In[22]:


get_ipython().run_cell_magic('time', '', "logit_c_values1 = np.logspace(-4, 2, 10)\n\nlogit_grid_searcher1 = LogisticRegressionCV(Cs = logit_c_values1, multi_class='multinomial'\n                                            , cv = skf, random_state=17, n_jobs=-1)\nlogit_grid_searcher1.fit(X_train, y_train)")


# In[23]:


def get_metrics_v2(inp_logit_c_values, inp_logit_grid_searcher):

    ret_df = pd.DataFrame(index = inp_logit_c_values, columns = [])

    score_list = []
    # class
    for idx, el in enumerate(list(inp_logit_grid_searcher.scores_.values())):
        ret_df['class' + str(idx) + str(0)] = el[0]
        ret_df['class' + str(idx) + str(1)] = el[1]
        ret_df['class' + str(idx) + str(2)] = el[2]

    ret_df['mean_train_score'] = ret_df.mean(axis = 1)
    ret_df['std_train_score'] = ret_df.std(axis = 1)
    
    return ret_df


# In[24]:


df = get_metrics_v2(logit_c_values1, logit_grid_searcher1)
df[['mean_train_score', 'std_train_score']]


# In[25]:


plt.figure(figsize = (15, 7))
plt.plot(logit_grid_searcher1.Cs_, df['mean_train_score']);


# In[26]:


print(max(df['mean_train_score']))
print(df[df['mean_train_score'] == max(df['mean_train_score'])].index)


# In[27]:


get_ipython().run_cell_magic('time', '', "logit_c_values2 = np.logspace(0.1, 7, 20)\n\nlogit_grid_searcher2 = LogisticRegressionCV(Cs = logit_c_values2, multi_class='multinomial'\n                                            , cv = skf, random_state=17, n_jobs=-1)\nlogit_grid_searcher2.fit(X_train, y_train)\n\ndf2 = get_metrics_v2(logit_c_values2, logit_grid_searcher2)\ndf2[['mean_train_score', 'std_train_score']]")


# In[28]:


print(max(df2['mean_train_score']))
print(df2[df2['mean_train_score'] == max(df2['mean_train_score'])].index)


# In[29]:


plt.figure(figsize = (15, 7))
plt.plot(logit_grid_searcher2.Cs_, df2['mean_train_score']);


# In[30]:


logit_cv_acc = accuracy_score(y_valid, logit_grid_searcher2.predict(X_valid))
logit_cv_acc


# In[31]:


answ3 = [round(max(df2['mean_train_score']), 2), round(logit_cv_acc, 2)]
answ3


# In[32]:


write_answer_to_file(answ3, 'answer4_3.txt')


# In[ ]:





# LinearSVC

# In[31]:


clf_inp = LinearSVC(C = 1, random_state = 17)
acc_cv, acc_val = get_metrics(clf_inp, skf, X_train, y_train, X_valid, y_valid)
acc_cv, acc_val


# In[32]:


get_ipython().run_cell_magic('time', '', "svm_params1 = {'C': np.linspace(1e-4, 1e4, 10)}\n\nsvm_grid_searcher1 = GridSearchCV(LinearSVC(random_state = 17), svm_params1, n_jobs=-1, return_train_score = True)\nsvm_grid_searcher1.fit(X_train, y_train)")


# In[33]:


print(svm_grid_searcher1.best_score_)
print(svm_grid_searcher1.best_params_)


# In[34]:


print(max(svm_grid_searcher1.cv_results_['mean_train_score']))
print(svm_grid_searcher1.cv_results_['mean_train_score'][7])
print(svm_grid_searcher1.cv_results_['params'][7])


# In[35]:


plot_validation_curves(svm_params1['C'], svm_grid_searcher1.cv_results_)


# In[36]:


get_ipython().run_cell_magic('time', '', "svm_params2 = {'C': np.linspace(1e-3, 1, 30)}\n\nsvm_grid_searcher2 = GridSearchCV(LinearSVC(random_state = 17), svm_params2, n_jobs=-1, return_train_score = True)\nsvm_grid_searcher2.fit(X_train, y_train)")


# In[37]:


print(svm_grid_searcher2.best_score_)
print(list(svm_grid_searcher2.best_params_.values())[0])


# In[38]:


plot_validation_curves(svm_params2['C'], svm_grid_searcher2.cv_results_)


# In[39]:


clfSVC = LinearSVC(C = list(svm_grid_searcher2.best_params_.values())[0], random_state=17)
clfSVC.fit(X_train, y_train)
svm_cv_acc = accuracy_score(y_valid, clfSVC.predict(X_valid))
svm_cv_acc


# In[40]:


print(max(svm_grid_searcher2.cv_results_['mean_train_score']))
print(svm_grid_searcher2.cv_results_['mean_train_score'][4])
print(svm_grid_searcher2.cv_results_['params'][4])


# In[41]:


svc_acc_cv, svc_acc_val = get_metrics(LinearSVC(C = list(svm_grid_searcher2.best_params_.values())[0], random_state = 17), skf, X_train, y_train, X_valid, y_valid)
svc_acc_cv, svc_acc_val


# In[42]:


answ4_4_2 = svc_acc_val


# In[43]:


answ4_4_1 = np.mean(svc_acc_cv)#svm_cv_acc#accuracy_score(y_valid, svm_grid_searcher2.predict(X_valid))


# In[44]:


answ4_4 = [round(answ4_4_1, 2), round(answ4_4_2, 2)]
answ4_4


# In[45]:


write_answer_to_file(answ4_4, 'answer4_4.txt')


# In[ ]:





# In[ ]:





# # Часть 2. Выбор параметров – длины сессии и ширины окна

# In[ ]:





# In[46]:


def model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv, random_state=17, test_size=0.3):
    '''
    Estimates CV-accuracy for (1 - test_size) share of (X_sparse, y) 
    loaded from path_to_X_pickle and path_to_y_pickle and holdout accuracy for (test_size) share of (X_sparse, y).
    The split is made with stratified train_test_split with params random_state and test_size.
    
    :param estimator – Scikit-learn estimator (classifier or regressor)
    :param path_to_X_pickle – path to pickled sparse X (instances and their features)
    :param path_to_y_pickle – path to pickled y (responses)
    :param cv – cross-validation as in cross_val_score (use StratifiedKFold here)
    :param random_state –  for train_test_split
    :param test_size –  for train_test_split
    
    :returns mean CV-accuracy for (X_train, y_train) and accuracy for (X_valid, y_valid) where (X_train, y_train)
    and (X_valid, y_valid) are (1 - test_size) and (testsize) shares of (X_sparse, y).
    '''
    
    with open(path_to_X_pickle, 'rb') as X_pkl:
        X_inner = pickle.load(X_pkl)
    with open(path_to_y_pickle, 'rb') as y_pkl:
        y_inner = pickle.load(y_pkl)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_inner, y_inner, 
                                                      test_size=test_size, 
                                                     random_state=random_state, stratify=y_inner)
    #skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    ret_acc_cv = []
    #for train_index, test_index in skf.split(X_train, y_train):
    for train_index, test_index in cv.split(X_train, y_train):
        clf = estimator
        clf.fit(X_train[train_index], y_train[train_index])
        predicted = clf.predict(X_train[test_index])
        ret_acc_cv.append(accuracy_score(y_train[test_index], predicted))
        
    clf = estimator
    clf.fit(X_train, y_train)
    ret_acc_val = accuracy_score(y_valid, clf.predict(X_valid))
    
    return np.mean(ret_acc_cv), ret_acc_val


# In[47]:


model_assessment(svm_grid_searcher2.best_estimator_
                 , os.path.join(PATH_TO_DATA, 'X_sp_10users.pkl')
                 , os.path.join(PATH_TO_DATA, 'y_10users.pkl')
                 , skf, random_state=17, test_size=0.3)


# In[48]:


get_ipython().run_cell_magic('time', '', "estm = svm_grid_searcher2.best_estimator_\nskf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)\nresults = {}\n\nfor window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):\n    if window_size <= session_length:\n        path_to_X_pkl = os.path.join(PATH_TO_DATA, f'X_sparse_10users_s{session_length}_w{window_size}.pkl')\n        #path_to_y_pkl = os.path.join(PATH_TO_DATA, f'Y_sparse_10users_s{session_length}_w{window_size}.pkl')\n        path_to_y_pkl = os.path.join(PATH_TO_DATA, f'Y_sparse_10users_s{session_length}_w{window_size}.pkl')\n        #print(path_to_X_pkl)\n        acc_cv, acc_val = model_assessment(svm_grid_searcher2.best_estimator_ \n                         , path_to_X_pkl\n                         , path_to_y_pkl \n                         , cv = skf, random_state=17, test_size=0.3)\n        results[f's{session_length}_w{window_size}'] = (acc_cv, acc_val)")


# In[49]:


results


# In[50]:


answ4_5_1 = results['s15_w5'][0]
answ4_5_2 = results['s15_w5'][1]
answ4_5_1, answ4_5_2


# In[51]:


answ4_5 = [round(answ4_5_1, 2), round(answ4_5_2, 2)]
answ4_5


# In[52]:


write_answer_to_file(answ4_5, 'answer4_5.txt')


# In[ ]:





# In[63]:


get_ipython().run_cell_magic('time', '', "estm = svm_grid_searcher2.best_estimator_\nskf  = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)\nresults150 = {}\n\nfor window_size, session_length in tqdm(itertools.product([10, 7, 5], [15, 10, 7, 5])):\n    if window_size <= session_length:\n        path_to_X_pkl = os.path.join(PATH_TO_DATA, f'X_sparse_150users_s{session_length}_w{window_size}.pkl')\n        #path_to_y_pkl = os.path.join(PATH_TO_DATA, f'Y_sparse_10users_s{session_length}_w{window_size}.pkl')\n        path_to_y_pkl = os.path.join(PATH_TO_DATA, f'Y_sparse_150users_s{session_length}_w{window_size}.pkl')\n        #print(path_to_X_pkl)\n        acc_cv, acc_val = model_assessment(svm_grid_searcher2.best_estimator_ \n                         , path_to_X_pkl\n                         , path_to_y_pkl \n                         , cv = skf, random_state=17, test_size=0.3)\n        results150[f's{session_length}_w{window_size}'] = (acc_cv, acc_val)")


# In[64]:


results150

{'s15_w10': (0.550363371512566, 0.5766798034350217),
 's10_w10': (0.4616145225656354, 0.48447915146207365),
 's15_w7': (0.5842562089277951, 0.6113022447797799),
 's10_w7': (0.502628509268666, 0.5261266478182376),
 's7_w7': (0.4374078763743305, 0.45415062847021154),
 's15_w5': (0.6165801657473978, 0.6391930695478659),
 's10_w5': (0.5275663749787812, 0.5487905776999076),
 's7_w5': (0.46695200044261503, 0.4830510536818027),
 's5_w5': (0.4093985555889958, 0.42267727648805176)}
# Как мы видим в обоих выриантах лучшее решение с самой длинной сессией.    
# Ширина окна скорее всего сказывается на объеме сессий в обучающей выборке.

# In[65]:


answ4_6_1, answ4_6_2 = results150['s10_w10']


# In[66]:


answ4_6 = [round(answ4_6_1, 2), round(answ4_6_2, 2)]
answ4_6


# In[67]:


write_answer_to_file(answ4_6, 'answer4_6.txt')


# In[ ]:





# # Часть 3. Идентификация  конкретного пользователя и кривые обучения

# In[68]:


with open(os.path.join(PATH_TO_DATA, 'X_sp_150users.pkl'), 'rb') as X_sparse_150users_pkl:
     X_sparse_150users = pickle.load(X_sparse_150users_pkl)
with open(os.path.join(PATH_TO_DATA, 'y_150users.pkl'), 'rb') as y_150users_pkl:
    y_150users = pickle.load(y_150users_pkl)
    
X_train_150, X_valid_150, y_train_150, y_valid_150 = train_test_split(X_sparse_150users, 
                                                                      y_150users, test_size=0.3, 
                                                     random_state=17, stratify=y_150users)


# In[114]:


get_ipython().run_cell_magic('time', '', "warnings.filterwarnings('ignore')\n\nlogit_cv_150users = LogisticRegressionCV(Cs =[logit_grid_searcher2.Cs_[10]], multi_class='ovr'\n                                            , cv = skf, random_state=17, n_jobs=-1)\nlogit_cv_150users.fit(X_train_150, y_train_150)\n\nwarnings.filterwarnings('default')")


# In[127]:


cv_scores_by_user = {}
for user_id in logit_cv_150users.scores_:
    cv_scores_by_user[user_id] = np.mean([float(logit_cv_150users.scores_[user_id][0])                                        , float(logit_cv_150users.scores_[user_id][1])                                        , float(logit_cv_150users.scores_[user_id][2])]
                                        , axis = 0
                                        )
    print(f'User {user_id}, CV score: {cv_scores_by_user[user_id]}')


# разницу между долей правильных ответов на кросс-валидации (только что посчитанную с помощью LogisticRegressionCV) и 
# долей меток в y_train_150, отличных от ID этого пользователя (именно такую долю правильных ответов можно получить, если классификатор всегда "говорит", что это не пользователь с номером 𝑖 в задаче классификации 𝑖-vs-All).

# In[137]:


class_distr = np.bincount(y_train_150.astype('int'))
acc_diff_vs_constant = []

for user_id in np.unique(y_train_150):
    same = 0
    for el in y_train_150:
        if el == user_id:
            same += 1
            
    #print(1 - same/y_train_150.shape[0])
    acc_diff_vs_constant.append(cv_scores_by_user[user_id] - (1 - same/y_train_150.shape[0]))


# In[138]:


num_better_than_default = (np.array(acc_diff_vs_constant) > 0).sum()
num_better_than_default


# In[130]:


answ4_7 = round(1 - num_better_than_default/len(np.unique(y_train_150)), 2)
answ4_7


# In[121]:


write_answer_to_file(answ4_7, 'answer4_7.txt')


# In[ ]:





# In[ ]:





# In[154]:


y_binary_128 = [int(el == 128) for el in y_train_150]


# In[156]:


sum(y_binary_128)


# In[158]:


def plot_learning_curve(val_train, val_test, train_sizes, 
                        xlabel='Training Set Size', ylabel='score'):
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                         facecolor=lines[0].get_color(), alpha=0.2)
    plot_with_err(train_sizes, val_train, label='train')
    plot_with_err(train_sizes, val_test, label='valid')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(loc='lower right');


# In[159]:


get_ipython().run_cell_magic('time', '', 'train_sizes = np.linspace(0.25, 1, 20)\nestimator = svm_grid_searcher2.best_estimator_\nn_train, val_train, val_test = learning_curve(estimator, X_train_150, y_train_150,\n                                             train_sizes = train_sizes\n                                             , cv = skf, random_state=om_state = 17\n                                             , n_jobs = -1\n                                             )')


# In[160]:


plot_learning_curve(val_train, val_test, n_train, 
                    xlabel='train_size', ylabel='accuracy')


# In[ ]:




