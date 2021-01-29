#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import os
import glob
import pickle
from collections import Counter

import re
from tqdm import tqdm


# In[2]:


DATA = './data'
MY_DATA = './my_data'
ANSW = './answers'


# In[3]:


tmp_data = pd.read_csv(os.path.join(DATA, '10users/user0128.csv'))
tmp_data.shape


# In[4]:


tmp_data.head()


# In[5]:


def prepare_train_set(path_to_csv, sess_len = 10):
    col_names = ['site' + str(n+1) for n in range(sess_len)] + ['user_id']
    #ret_data = pd.DataFrame(columns = col_names)
    freq_site_dict = {}
    site_cnt = Counter()
    rows = 0

    #for file_csv in sorted(glob.glob(path_to_csv + '/*.csv')):
    for file_csv in tqdm(sorted(glob.glob(path_to_csv + '/*.csv'))):
        temp_dataframe = pd.read_csv(file_csv)
        site_cnt += Counter(temp_dataframe.site.values)
        rows += round(temp_dataframe.shape[0] / sess_len + 0.499)

    rows = rows
    #print(rows)

    for idx, site_dict in enumerate(site_cnt.most_common(), start = 1):
        freq_site_dict[site_dict[0]] = (idx, site_dict[1])


    ret_data = pd.DataFrame(index = range(rows), columns = col_names)
    index = 0

    for file_csv in tqdm(sorted(glob.glob(path_to_csv + '/*.csv'))):
        temp_dataframe = pd.read_csv(file_csv)
        userid = int(re.findall('\d+', file_csv)[1])
        #print(file_csv)    

        sess_numb = round(temp_dataframe.shape[0] / sess_len + 0.499)
        for idx in range(sess_numb - 1):
            #new_sess = {}
            new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}
            for n, site in enumerate(temp_dataframe.site.values[idx*sess_len: (idx + 1)*sess_len]):
                new_sess['site' + str(n+1)] = freq_site_dict[site][0]
            new_sess['user_id'] = userid

            #ret_data = ret_data.append(new_sess, ignore_index = True)
            ret_data.iloc[index] = new_sess
            index += 1

        new_sess = {('site' + str(n+1)):0 for n in range(sess_len)}
        for n, site in enumerate(temp_dataframe.site.values[(sess_numb-1)*sess_len: ]):
            new_sess['site' + str(n+1)] = freq_site_dict[site][0]
        new_sess['user_id'] = userid

        #ret_data = ret_data.append(new_sess, ignore_index = True)
        #print(new_sess)
        ret_data.iloc[index] = new_sess
        index += 1
        
    return ret_data.fillna(0).astype(np.int), freq_site_dict


# In[6]:


train_3user, site_freq_3users = prepare_train_set(os.path.join(DATA, '3users'))


# In[ ]:





# In[7]:


#0:35
train_10users, site_freq_10users = prepare_train_set(os.path.join(DATA, '10users'))


# In[8]:


def write_answer_to_file(answer, file_address):
    with open(os.path.join(ANSW, file_address), 'w') as out_f:
        out_f.write(str(answer))


# In[9]:


#answ1 = train_10users.drop_duplicates(subset = ['site' + str(n+1) for n in range(10)]).shape[0]
answ1 = train_10users.shape[0]
answ1


# In[10]:


write_answer_to_file(answ1, 'answer1_1.txt')


# In[11]:


answ2 = len(site_freq_10users)
answ2


# In[12]:


write_answer_to_file(answ2, 'answer1_2.txt')


# In[13]:


#51:30
train_150users, site_freq_150users = prepare_train_set(os.path.join(DATA, '150users'))


# In[14]:


#answ3 = train_150users.drop_duplicates(subset = ['site' + str(n+1) for n in range(10)]).shape[0]
answ3 = train_150users.shape[0]
answ3


# In[15]:


write_answer_to_file(answ3, 'answer1_3.txt')


# In[16]:


answ4 = len(site_freq_150users)
answ4


# In[17]:


write_answer_to_file(answ4, 'answer1_4.txt')


# In[ ]:





# In[18]:


i = 0
with open(os.path.join(ANSW, 'answer1_5.txt'), 'w') as out_f:
    for site in site_freq_150users:
        out_f.write(site + ' ')
        i += 1
        if i == 10:
            break
            
#delete last ' '


# In[ ]:





# In[19]:


train_10users.to_csv(os.path.join(MY_DATA, 'train_data_10users.csv'), 
                        index_label='session_id', float_format='%d')
train_150users.to_csv(os.path.join(MY_DATA, 'train_data_150users.csv'), 
                         index_label='session_id', float_format='%d')


# In[20]:


train_10users.values.max(), train_150users.values.max()


# In[21]:


def to_sitebow_csr(inp_data):
    row = []
    col = []
    data = []
    for idx, elmnt in enumerate(inp_data):
        cnt = Counter(elmnt)
        
        row += [idx] * len(cnt)
        data+= list(cnt.values())
        col += list(cnt.keys())

    ret = csr_matrix((data, (row, col)), shape = (max(row)+1, max(col)+1))[:, 1:]
    
    return ret


# In[22]:


to_sitebow_csr(train_3user.iloc[:, :-1].values).toarray()


# In[23]:


train_10users = pd.read_csv(os.path.join(MY_DATA, 'train_data_10users.csv'), index_col='session_id')
train_150users = pd.read_csv(os.path.join(MY_DATA, 'train_data_150users.csv'), index_col='session_id')
train_10users.values.max(), train_150users.values.max()


# In[24]:


print(len(site_freq_3users))
print(len(site_freq_10users))
print(len(site_freq_150users))


# In[25]:


x_3users, y_3users = train_3user.iloc[:, :-1].values, train_3user.iloc[:, -1].values
x_10users, y_10users = train_10users.iloc[:, :-1].values, train_10users.iloc[:, -1].values
x_150users, y_150users = train_150users.iloc[:, :-1].values, train_150users.iloc[:, -1].values


# In[26]:


print(x_3users.shape)
print(x_10users.shape)
print(x_150users.shape)

X_sp_3users = to_sitebow_csr(x_3users)
X_sp_10users = to_sitebow_csr(x_10users)
X_sp_150users = to_sitebow_csr(x_150users)

print(X_sp_3users.shape)
print(X_sp_10users.shape)
print(X_sp_150users.shape)


# In[27]:


X_sp_3users.toarray()


# In[28]:


with open(os.path.join(MY_DATA, 'X_sp_3users.pkl'), 'wb') as X3_pkl:
    pickle.dump(X_sp_3users, X3_pkl, protocol = 2)
    
with open(os.path.join(MY_DATA, 'X_sp_10users.pkl'), 'wb') as X10_pkl:
    pickle.dump(X_sp_10users, X10_pkl, protocol = 2)

with open(os.path.join(MY_DATA, 'X_sp_150users.pkl'), 'wb') as X150_pkl:
    pickle.dump(X_sp_150users, X150_pkl, protocol = 2)


# In[29]:


with open(os.path.join(MY_DATA, 'y_3users.pkl'), 'wb') as Y3_pkl:
    pickle.dump(y_3users, Y3_pkl, protocol = 2)
    
with open(os.path.join(MY_DATA, 'y_10users.pkl'), 'wb') as Y10_pkl:
    pickle.dump(y_10users, Y10_pkl, protocol = 2)

with open(os.path.join(MY_DATA, 'y_150users.pkl'), 'wb') as Y150_pkl:
    pickle.dump(y_150users, Y150_pkl, protocol = 2)


# In[30]:


with open(os.path.join(MY_DATA, 'site_freq_3users.pkl'), 'wb') as freq3_pkl:
    pickle.dump(site_freq_3users, freq3_pkl, protocol = 2)
    
with open(os.path.join(MY_DATA, 'site_freq_10users.pkl'), 'wb') as freq10_pkl:
    pickle.dump(site_freq_10users, freq10_pkl, protocol = 2)

with open(os.path.join(MY_DATA, 'site_freq_150users.pkl'), 'wb') as freq150_pkl:
    pickle.dump(site_freq_150users, freq150_pkl, protocol = 2)


# In[ ]:





# In[ ]:





# In[ ]:




