# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:31:25 2016

@author: govind
"""

import pandas as pd
import os
import pyprind
import numpy as np
from csv import reader
import random

############## Data preparation ################# 
os.chdir('C:/Users/govin/Documents/Learning and Development/Miscellaneous/Chipy project/sentiment data')


#Read in training data of tweets

if 'train_list' not in locals():
    train_list=[]
    pbar = pyprind.ProgBar(1600000)
    print('Reading in training data')
    with open('training.1600000.processed.noemoticon.csv','r') as file:
        for line in reader(file):
            train_list.append(line)
            pbar.update()
        

full_data = pd.DataFrame(train_list)

random.seed(20)
#Reorder data
full_data = full_data.reindex(np.random.permutation(full_data.index))
full_data.iloc[:,0].value_counts()


#Read in test data of tweets

if 'test_list' not in locals():
    test_list=[]
    print('Reading in test data')
    with open('testdata.manual.2009.06.14.csv','r') as file:
        for line in reader(file):
            test_list.append(line)
    
    
test_data = pd.DataFrame(test_list)

os.chdir('C:/Users/govin/Documents/Learning and Development/Miscellaneous/Chipy project')

#Subset only necessary columns
full_data = full_data.iloc[:,[0,1,5]]
test_data = test_data.iloc[:,[0,1,5]]
#Rename columns
full_data.columns = ['Sentiment','ID','Tweet']
test_data.columns = ['Sentiment','ID','Tweet']



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

###Replace punctuations###
from SentiFunctions import strip_punctuation
full_data['Tweet'] = full_data['Tweet'].apply(strip_punctuation)
test_data['Tweet'] = test_data['Tweet'].apply(strip_punctuation)


#Replace negation words with not and convert to lower case
negations = ['cant','cannot','wont','doesnt','dont','wouldnt','shouldnt',
             'wasnt','couldnt','isnt']
             
from nltk import word_tokenize

def replace_negation(x):
    text = word_tokenize(x)
    ans = ' '.join([x.lower() if x not in negations  else 'not' for x in text ])
    return(ans)

full_data['Tweet'] = full_data['Tweet'].apply(replace_negation)
test_data['Tweet'] = test_data['Tweet'].apply(replace_negation)


######Stem words #####
##########Training data#######

from nltk.stem.porter import *
stemmer = PorterStemmer()

def stem_tweet(tweet):
    words = tweet.split()    
    tweet_new = ' '.join([stemmer.stem(x) for x in words])
    return(tweet_new)
    
full_data['Tweet_stemmed'] = full_data['Tweet'].apply(stem_tweet)

########Testing data##########
test_data['Tweet_stemmed'] = test_data['Tweet'].apply(stem_tweet)
test_data['Sentiment_2'] = pd.Categorical(test_data['Sentiment'],['0','4','2'])
test_data = test_data.sort('Sentiment_2')

test_data['Sentiment'].value_counts()#positive - 177,neutral-139,negative - 182
#Retain only tweets with positive and negative sentiment in test data
test_data_pos_neg = test_data.loc[np.array((test_data['Sentiment']=='0') | (test_data['Sentiment']=='4'))]
test_docs = test_data_pos_neg['Tweet_stemmed'].values
#########Try a bag of words model#####

train_test_tweets =  full_data['Tweet_stemmed']
train_test_tweets = train_test_tweets.append(test_data['Tweet_stemmed'])

####Make a word frequency table
from sklearn.feature_extraction.text import CountVectorizer

######Training data
count = CountVectorizer(stop_words = 'english')
docs = train_test_tweets.values
#Create a bag of words
bag = count.fit_transform(docs)        
###Get number of occurences of all words in the corpus
count_mat = bag.sum(axis=0)
count_df = pd.DataFrame(count_mat,columns = sorted(count.vocabulary_)).T
count_df.columns=['word_count']
####Use only words with more than 100 occurrences
words = count_df[count_df['word_count']>=50] ####9853 words
selected_words = list(words.index)
##get index of selected words
selected_words_dict = {x:count.vocabulary_.get(x) for x in selected_words}
selected_words_index = [count.vocabulary_.get(x) for x in selected_words]

##Get just counts for chosen words
full_df = bag[:,selected_words_index] ##Sparse matrix

##datasets without a validation set
X_train1 = full_df[:1600000,:]
X_test = full_df[1600000:1600359,]

##datasets with a validation set
X_train2 = full_df[:1120000,:]
X_valid = full_df[1120000:1600000,:]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train1 = le.fit_transform(full_data['Sentiment'])
y_train2 = le.transform(full_data['Sentiment'][:1120000])
y_valid = le.transform(full_data['Sentiment'][1120000:])
y_test = le.transform(test_data_pos_neg['Sentiment'])



#####Try logistic regression models between ridge and lasso######
l2penalties = [0.1,1,10,100,1000,1e10]#1e10 corresponds to unregulaarized logistic regression
l1penalties = [1,10,100,1000]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
###Validation for L2 penalty
for i in l2penalties:
    log_reg_model = LogisticRegression(penalty='l2',C=i)
    log_reg_model.fit(X_train2,y_train2)
    preds = log_reg_model.predict(X_valid)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l2_penalty = %0.1f): %0.3f" %(i,accuracy))
 
##All have same accuracy of 0.769 i validation data set 
 ###########Train on full dataset###########
##Model training
from sklearn.linear_model import LogisticRegression

log_reg_ridge = LogisticRegression(penalty='l2',C = 10)
log_reg_ridge.fit(X_train1,y_train1)
preds = log_reg_ridge.predict(X_train1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_ridge.predict(X_test)
accuracy_score(y_test,test_preds)##80.5% accuracy
 
 ###Validation for l1 penalty
for i in l1penalties:
    log_reg_model = LogisticRegression(penalty='l1',C=i)
    log_reg_model.fit(X_train2,y_train2)
    preds = log_reg_model.predict(X_valid)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l1_penalty = %0.1f): %0.3f" %(i,accuracy))    
##Accuracy is the same for all l1-penalties
###############Train on full data set#############
##Model training
from sklearn.linear_model import LogisticRegression

log_reg_lasso = LogisticRegression(penalty='l1',C = 1)
log_reg_lasso.fit(X_train1,y_train1)
preds = log_reg_lasso.predict(X_train1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_lasso.predict(X_test)
accuracy_score(y_test,test_preds)##79.66% accuracy
 
    
######Try Binomial Naive Bayes Model with word stemming######
from sklearn.naive_bayes import BernoulliNB
##Convert word frequency matrix into binary matrix
X_train1_bin = X_train1.copy()
X_train1_bin[X_train1_bin>0] = 1


clf_ber_bayes = BernoulliNB()
clf_ber_bayes.fit(X_train1_bin,y_train1)

train_preds = clf_ber_bayes.predict(X_train1)
accuracy_score(y_train1,train_preds)
#Convert test dataframe to binary
X_test_bin = X_test.copy()
X_test_bin[X_test_bin>0]=1

test_preds = clf_ber_bayes.predict(X_test_bin)
accuracy_score(y_test,test_preds)##81.89 % accuracy_score


##############Try multinomial Bayes model with word stemming###

from sklearn.naive_bayes import MultinomialNB
clf_mul_bayes = MultinomialNB()
clf_mul_bayes.fit(X_train1,y_train1)

train_preds = clf_mul_bayes.predict(X_train1)
accuracy_score(y_train1,train_preds)


test_preds = clf_mul_bayes.predict(X_test)
accuracy_score(y_test,test_preds)##81.61% accuracy

############################try bag of words without word stemming###############

#########Try a bag of words model#####

train_test_tweets =  full_data['Tweet']
train_test_tweets = train_test_tweets.append(test_data['Tweet'])

from SentiFunctions import strip_punctuation
train_test_tweets = train_test_tweets.apply(strip_punctuation)

####Make a word frequency table
from sklearn.feature_extraction.text import CountVectorizer

######Training data
count = CountVectorizer(stop_words = 'english')
docs = train_test_tweets.values
#Create a bag of words
bag = count.fit_transform(docs)        
###Get number of occurences of all words in the corpus
count_mat = bag.sum(axis=0)
count_df = pd.DataFrame(count_mat,columns = sorted(count.vocabulary_)).T
count_df.columns=['word_count']
####Use only words with more than 100 occurrences
words = count_df[count_df['word_count']>=50] ####12361 words
selected_words = list(words.index)
##get index of selected words
selected_words_dict = {x:count.vocabulary_.get(x) for x in selected_words}
selected_words_index = [count.vocabulary_.get(x) for x in selected_words]

##Get just counts for chosen words
full_df = bag[:,selected_words_index] ##Sparse matrix

##datasets without a validation set
X_train1 = full_df[:1600000,:]
X_test = full_df[1600000:1600359,]

##datasets with a validation set
X_train2 = full_df[:1120000,:]
X_valid = full_df[1120000:1600000,:]


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train1 = le.fit_transform(full_data['Sentiment'])
y_train2 = le.transform(full_data['Sentiment'][:1120000])
y_valid = le.transform(full_data['Sentiment'][1120000:])
y_test = le.transform(test_data_pos_neg['Sentiment'])



#####Try logistic regression models between ridge and lasso######
l2penalties = [0.1,1,10,100,1000,1e10]#1e10 corresponds to unregulaarized logistic regression
l1penalties = [1,10,100,1000]

###Validation for L2 penalty
for i in l2penalties:
    log_reg_model = LogisticRegression(penalty='l2',C=i)
    log_reg_model.fit(X_train2,y_train2)
    preds = log_reg_model.predict(X_valid)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l2_penalty = %0.1f): %0.3f" %(i,accuracy))
 
##All have same accuracy
 ###########Train on full dataset###########
##Model training
from sklearn.linear_model import LogisticRegression

log_reg_ridge = LogisticRegression(penalty='l2',C = 1)
log_reg_ridge.fit(X_train1,y_train1)
preds = log_reg_ridge.predict(X_train1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_ridge.predict(X_test)
accuracy_score(y_test,test_preds)##82.45% accuracy
 
 ###Validation for l1 penalty
for i in l1penalties:
    log_reg_model = LogisticRegression(penalty='l1',C=i)
    log_reg_model.fit(X_train2,y_train2)
    preds = log_reg_model.predict(X_valid)
    accuracy = accuracy_score(y_valid,preds)
    print("Accuracy(l1_penalty = %0.1f): %0.3f" %(i,accuracy))    
##Accuracy is the same for all l1-penalties
###############Train on full data set#############
##Model training
from sklearn.linear_model import LogisticRegression

log_reg_lasso = LogisticRegression(penalty='l1',C = 1)
log_reg_lasso.fit(X_train1,y_train1)
preds = log_reg_lasso.predict(X_train1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_lasso.predict(X_test)
accuracy_score(y_test,test_preds)##81.89% accuracy

  
######Try Binomial Naive Bayes Model without word stemming######
from sklearn.naive_bayes import BernoulliNB
##Convert word frequency matrix into binary matrix
X_train1_bin = X_train1.copy()
X_train1_bin[X_train1_bin>0] = 1


clf_ber_bayes = BernoulliNB()
clf_ber_bayes.fit(X_train1_bin,y_train1)

train_preds = clf_ber_bayes.predict(X_train1_bin)
accuracy_score(train_preds,y_train1)
#Convert test dataframe to binary
X_test_bin = X_test.copy()
X_test_bin[X_test_bin>0]=1

test_preds = clf_ber_bayes.predict(X_test_bin)
accuracy_score(y_test,test_preds)##84.12 % accuracy_score


##############Try multinomial Bayes model without word stemming###

from sklearn.naive_bayes import MultinomialNB
clf_mul_bayes = MultinomialNB()
clf_mul_bayes.fit(X_train1,y_train1)


train_preds = clf_ber_bayes.predict(X_train1)
accuracy_score(train_preds,y_train1)

test_preds = clf_mul_bayes.predict(X_test)
accuracy_score(y_test,test_preds)##83.56% accuracy




############################Try Logistic Regression with tfidf################

train_tweets = full_data['Tweet']
##Remove punctuations



####Make a word frequency table
from sklearn.feature_extraction.text import CountVectorizer

######Training data
count = CountVectorizer(stop_words = 'english')

train_test_tweets =  full_data['Tweet']
train_test_tweets = train_test_tweets.append(test_data['Tweet'])
docs = train_test_tweets.values
#Create a bag of words
bag = count.fit_transform(docs) 

###Get number of occurences of all words in the corpus
count_mat = bag.sum(axis=0)
count_df = pd.DataFrame(count_mat,columns = sorted(count.vocabulary_)).T
count_df.columns=['word_count']
####Use only words with more than 100 occurrences
words = count_df[count_df['word_count']>=100] ####6600 words
selected_words = list(words.index)
##get index of selected words
selected_words_dict = {x:count.vocabulary_.get(x) for x in selected_words}
selected_words_index = [count.vocabulary_.get(x) for x in selected_words]

##Get just counts for chosen words
full_df = bag[:,selected_words_index] ##Sparse matrix

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X_train = full_df[:1600000,:]
X_test = full_df[1600000:1600359,:]

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

##Model training
from sklearn.linear_model import LogisticRegression

log_reg_lasso = LogisticRegression(penalty='l2',C = 1)
log_reg_lasso.fit(X_train_tfidf,y_train1)
preds = log_reg_lasso.predict(X_train_tfidf)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_lasso.predict(X_test_tfidf)
accuracy_score(y_test,test_preds)##82.72s% accuracy


#################Combine bag of words with senti features#############

train_mat_final = sparse.hstack([X_train1,senti_matrix_sparse])
test_mat_final = sparse.hstack([X_test,senti_sp_test_matrix])

##Model training
from sklearn.linear_model import LogisticRegression

log_reg_lasso = LogisticRegression(penalty='l1',C = 10)
log_reg_lasso.fit(train_mat_final,y_train1)
preds = log_reg_lasso.predict(train_mat_final)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

accuracy_score(y_train1,preds)

##Model testing
test_preds = log_reg_lasso.predict(test_mat_final)
accuracy_score(y_test,test_preds)##84.12% accuracy

















