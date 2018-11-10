# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:44:23 2016

@author: govin
"""

import pandas as pd
import os
import pyprind
import numpy as np
from csv import reader
import random
import itertools
from scipy import sparse

os.chdir('C:/Users/govin/Documents/Learning and Development/Miscellaneous/Chipy project')
###Cleaning###


##Replace usernames and weblinks
def replace_usernames_links(x):
    temp = [word  if not(word.startswith("http")) else "URL" for word in x.split()]
    temp = [word if not(word.startswith("@")) else "UserName" for word in temp]
    return(' '.join(temp))

full_data['Tweet'] = full_data['Tweet'].apply(replace_usernames_links)
test_data['Tweet'] = test_data['Tweet'].apply(replace_usernames_links)

##Replacing digits
full_data['Tweet'] = full_data['Tweet'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))
test_data['Tweet'] = test_data['Tweet'].apply(lambda x: ''.join(i for i in x if not i.isdigit()))

###Replacing words with multiple letters such as hungrrryyyy to hungrryy
import re

full_data['Tweet'] = full_data['Tweet'].apply(lambda x: re.sub(r'(.)\1{2,}',r'\1\1\1',x))
test_data['Tweet'] = test_data['Tweet'].apply(lambda x: re.sub(r'(.)\1{2,}',r'\1\1\1',x))


##Replacing '&quot;','&amp;' and anything similar
full_data['Tweet'] = full_data['Tweet'].apply(lambda x: re.sub(r'&.+?;','',x))
test_data['Tweet'] = test_data['Tweet'].apply(lambda x: re.sub(r'&.+?;','',x))

test_data['Sentiment_2'] = pd.Categorical(test_data['Sentiment'],['0','4','2'],ordered=True)
test_data = test_data.sort_values(by='Sentiment_2')

test_data['Sentiment'].value_counts()#positive - 177,neutral-139,negative - 182

test_data_pos_neg = test_data.loc[np.array((test_data['Sentiment']=='0') | (test_data['Sentiment']=='4'))]
#######Try a model with just senti features
from SentiFunctions import get_features
from SentiFunctions import get_previous_word
from tqdm import tqdm, tqdm_pandas
tqdm.pandas()


senti_df = full_data['Tweet'][:20000].progress_apply(get_features) #How to parallelize this?
senti_test_df = test_data['Tweet'].apply(get_features)

import itertools
senti_test_matrix = np.array(list(itertools.chain(*senti_test_df.values))).reshape(senti_test_df.shape[0],24)
senti_matrix = np.array(list(itertools.chain(*senti_df.values))).reshape(senti_df.shape[0],24)


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train1 = le.fit_transform(full_data['Sentiment'][:20000])
y_train2 = le.transform(full_data['Sentiment'][:1120000])
y_valid = le.transform(full_data['Sentiment'][1120000:])
y_test = le.transform(test_data_pos_neg['Sentiment'])


senti_matrix_all = np.column_stack((senti_matrix,y_train1))
##Save to file
np.savetxt('senti_matrix_all.csv',senti_matrix,delimiter=",")
np.savetxt('senti_test_matrix.csv',senti_test_matrix,delimiter=",")

##Read file from disk
senti_matrix = np.loadtxt(open('senti_matrix_all.csv','rb'),delimiter=",")
senti_test_matrix = np.loadtxt(open('senti_test_matrix.csv','rb'),delimiter=",")

senti_matrix_sparse =sparse.csr_matrix(senti_matrix)
senti_train_matrix = senti_matrix[:1120000,:]
senti_sp_train_matrix = sparse.csc_matrix(senti_train_matrix)
senti_valid_matrix = senti_matrix[1120000:,:]
senti_sp_valid_matrix = sparse.csr_matrix(senti_valid_matrix)

#Subset only positive and negative tweets
senti_test_matrix2 = senti_test_matrix[:359,:]
senti_sp_test_matrix = sparse.csr_matrix(senti_test_matrix2)

#####Create smaller training datasets for advanced algos
senti_train_matrix2 = senti_matrix[:15000,:]
senti_valid_matrix2 = senti_matrix[15000:20000,:]

####Target vectors for smaller dataset
y_train_small = le.transform(full_data['Sentiment'][:15000])
y_valid_small = le.transform(full_data['Sentiment'][15000:20000])

###Create datasets for full scale training
senti_train_matrix3 = senti_matrix[:20000,:]
y_train3 = le.transform(full_data['Sentiment'][:20000])

from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


#####Try logistic regression models between ridge and lasso######
l2penalties = [0.1,1,10,100,1000,1e10]#1e10 corresponds to unregularized logistic regression
l1penalties = [1,10,100,1000]

###Validation for L2 penalty
for i in l2penalties:
    log_reg_model = LogisticRegression(penalty='l2',C=i)
    log_reg_model.fit(senti_train_matrix,y_train2)
    preds = log_reg_model.predict(senti_valid_matrix)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l2_penalty = %0.1f): %0.3f" %(i,accuracy))
 
##Same accuracy for all l2 penalties
 
senti_log_reg = LogisticRegression(penalty='l2',C=10)
##Train on full dataset
senti_log_reg.fit(senti_matrix,y_train1)
preds = senti_log_reg.predict(senti_matrix)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train1,preds)
accuracy_score(y_train1,preds)#66.69

#Model testing
test_preds = senti_log_reg.predict(senti_test_matrix2)
confusion_matrix(y_test,test_preds)
accuracy_score(y_test,test_preds)#71.86% 
 
 
 ###Validation for l1 penalty
for i in l1penalties:
    log_reg_model = LogisticRegression(penalty='l1',C=i)
    log_reg_model.fit(senti_train_matrix,y_train2)
    preds = log_reg_model.predict(senti_valid_matrix)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l1_penalty = %0.1f): %0.3f" %(i,accuracy))    
##Accuracy is same for all l1_penalties

senti_log_reg = LogisticRegression(penalty='l1',C=10)
##Train on full dataset
senti_log_reg.fit(senti_matrix,y_train1)
preds = senti_log_reg.predict(senti_matrix)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train1,preds)
accuracy_score(y_train1,preds)#66.69

#Model testing
test_preds = senti_log_reg.predict(senti_test_matrix2)
confusion_matrix(y_test,test_preds)
accuracy_score(y_test,test_preds)#71.86% 
 

#########################SVMs take too much time to train#########################
##################Try SVM with senti features#######
#SVM with linear kernel 
from sklearn.svm import SVC

C = [0.01,0.1,1,10,100]

for c in C:
    lin_svm_model = SVC(kernel= 'linear',C=c,random_state=0)
    lin_svm_model.fit(senti_train_matrix2,y_train_small)
    preds = lin_svm_model.predict(senti_valid_matrix2)
    accuracy = accuracy_score(y_valid_small,preds)
    print("Accuracy(C = %0.2f): %0.3f" %(c,accuracy))
#C=10 has best accuracy of 0.672

lin_svm_best = SVC(kernel= 'linear',C=10,random_state=100)
lin_svm_best.fit(senti_train_matrix3,y_train3)

train_preds = lin_svm_best.predict(senti_train_matrix3)
accuracy_score(y_train3,train_preds)#0.6694

preds = lin_svm_best.predict(senti_test_matrix2)
accuracy_svm_lin = accuracy_score(y_test,preds)#0.6935
print(accuracy_svm_lin)

#SVM with a radial basis kernel

C = [0.1,1,10,100]
gamma = [0.1,1,10]

for c in C:
    for g in gamma:
        svm_radial = SVC(kernel='rbf',random_state=0,gamma=g,C=c)
        svm_radial.fit(senti_train_matrix2,y_train_small)
        preds = svm_radial.predict(senti_valid_matrix2)
        accuracy = accuracy_score(y_valid_small,preds)
        print("Accuracy(C = %0.2f,gamma = %0.2f): %0.3f" %(c,g,accuracy))
###Best performance when C =1 and gamma =0.1-> accuracy =0.637
        

svm_radial_best = SVC(kernel='rbf',random_state=0,gamma=0.1,C=1)
svm_radial_best.fit(senti_train_matrix3,y_train3)

train_preds = svm_radial_best.predict(senti_train_matrix3)
accuracy_score(y_train3,train_preds)0.8505

preds = svm_radial_best.predict(senti_test_matrix2)
accuracy = accuracy_score(y_test,preds)
print(accuracy)##accuracy =0.6211

##############Try Adaboost########
#==============================================================================
 from sklearn.ensemble import AdaBoostClassifier
 from sklearn.tree import DecisionTreeClassifier
 tree = DecisionTreeClassifier(criterion = 'entropy',
                               max_depth=1,
                               random_state=0)
 
 ada = AdaBoostClassifier(base_estimator = tree,
                          n_estimators = 1000,
                          learning_rate=0.1,
                          random_state=0)
 
 
 ada = ada.fit(senti_train_matrix3,y_train3)
 y_train_pred = ada.predict(senti_train_matrix3)
 print(accuracy_score(y_train3,y_train_pred))##0.671 on training data
 y_test_pred =ada.predict(senti_test_matrix2)
 print(accuracy_score(y_test,y_test_pred)) ##68.8 % accouracy
#==============================================================================

#################Try Random forest####################

from sklearn.ensemble import RandomForestClassifier
feat_labels = np.array(['exclm_flag','noofhashtags','no_of_words','capwords','cap_letter_perc','NoofURL',
               'allcapwords','tweet_len','negation_words','no_polar_words','no_non_polar_words',
               'polar_nouns','non_polar_nouns','polar_verbs','non_polar_verbs','polar_adverbs',
               'non_polar_adverbs','polar_adj','non_polar_adj','polarity_nouns','polarity_verbs',
                     'polarity_adj','polarity_adv','polarity_total'])

forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)
forest.fit(senti_train_matrix3,y_train3)


y_train_preds = forest.predict(senti_train_matrix3)
accuracy_score(y_train3,y_train_preds) #0.9943

y_test_preds = forest.predict(senti_test_matrix2)
accuracy_score(y_test,y_test_preds) #0.7047

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(24):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

from matplotlib import pyplot as plt
plt.title('Feature Importances')
plt.bar(range(24),importances[indices],color='lightblue',align='center')
plt.xticks(range(24),feat_labels[indices],rotation=90)
plt.xlim(-1,24)
plt.tight_layout()
plt.show()


########################Try gradient boosting#############
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators = 100,learning_rate =1,max_depth=1,
                                 random_state=0)
clf.fit(senti_train_matrix3,y_train3)


y_train_preds = clf.predict(senti_train_matrix3)
accuracy_score(y_train3,y_train_preds) #0.6866


y_test_preds = clf.predict(senti_test_matrix2)
accuracy_score(y_test,y_test_preds) #0.6824


