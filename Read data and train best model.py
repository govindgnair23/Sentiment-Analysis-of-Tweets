# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:29:18 2017

@author: govin
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
####Use only words with more than 50 occurrences
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