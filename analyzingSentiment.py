#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:05:05 2020

@author: Ngo Quoc Khanh
@ Gmail: khanh.ngoakatekhanh@hcmut.edu.vn
@install package using anaconda: conda install -c anaconda <package>
"""
import time
import os
import numpy
from itertools import islice
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



'''
View data
product/productId: B001E4KFG0
review/userId: A3SGXH7AUHU8GW
review/profileName: delmartian
review/helpfulness: 1/1
review/score: 5.0
review/time: 1303862400
review/summary: Good Quality Dog Food
review/text: I have bought several of the Vitalit
'''


def time_count(start, end):
    diff = end - start
    mins = round(diff/60, 2)
    # round function
    if diff > 60:
        secons = round(diff%60,2)
    else:
        secons = 0
    print('It takes : ' + str(mins) + ' minutes ' + str(secons) + ' seconds.')

# Clean html in a sentence using BeaitufulSoup
def clean_html(sentence):
    review_text = BeautifulSoup(sentence, 'html.parser').get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    return letters_only

def text_to_csv(file_in, file_out):
    # time start
    t0 = time.time()
    
    with open(file_in, 'r', encoding='ISO-8859-1') as f1, open(file_out, 'w') as f2:
        i = 0
        # Product Id, score , summary, review_text
        f2.write('ProductId,Score,Summary,Text\n')
        
        while True: 
            n_lines = list(islice(f1, 9))
            if not n_lines:
                break # end of file
            write_out = ''
            for line in n_lines:
                # Split the string into 2 parts and trim the space
                if 'product/productId' in line:
                    product = line.split(':')[1].strip() + ','
                    write_out = write_out + product 
                elif 'review/score' in line:
                    score = line.split(':')[1].strip() + ','
                    write_out = write_out + score
                elif 'review/summary' in line:
                    summary = clean_html(line.split(':')[1].strip()) + ','
                    write_out = write_out + summary
                elif 'review/text' in line:
                    text = clean_html(line.split(':')[1].strip()) + '\n'
                    write_out = write_out + text
                
            # Write data here
            f2.write(write_out)
            i += 1
            # Print the status of the process
            if(i % 30000 == 0):
                print('Removed html tags: ' + str(i))
        
        # Print time take
        time_count(t0,time.time())
    

# This line process data by remove html tags
#text_to_csv('finefoods.txt', 'temp_dummy.csv')

# Clean data: clean hmtl, lower case, remove stop word, write in the new file, using Bag of Words model

# REMOVE STOP WORDS , KEEP MEANING WORDS
#list_stop_words = stopwords.words('english')
def remove_stop_words(sentence):
    # convert to lower words
    words = sentence.lower().split()
    # Convert from list to set
    stop_words = set(stopwords.words('english'))
    meaningful_words = []
    # join the meaningful words together
    meaningful_words = [w for w in words if not w in stop_words]
    return ' '.join(meaningful_words)

# Define clean all stop words
#df = pd.read_csv('temp_dummy.csv')
def clean_all_stop_words(data_frame, out_file):
    t0 = time.time()
    removed_to_csv = []
    i = 0
    df_len = data_frame.shape[0]
    for row in data_frame.iterrows():
        str_product = row[1]['ProductId']
        str_score = row[1]['Score']
        str_summary = remove_stop_words(str(row[1]['Summary']))
        str_text = remove_stop_words(str(row[1]['Text']))
        removed_to_csv.append(str_product + ',' + str(str_score) + ',' + str_summary + ',' + str_text + '\n')
        if (i % 30000 == 0):
            print('Removed stop words ' + str(i) + ' of ' + str(df_len))
        i += 1
    print('Removed all stop words!!!\n')
    
    # list to csv
    i = 0
    with open(out_file, 'w') as f:
        f.write('ProductId,Score,Summary,Text\n')
        for line in removed_to_csv:
            f.write(line)
            if (i % 30000 == 0):
                print('Wtriting ' + str(i) + ' of ' + str(df_len))
            i += 1
        print('Done writting!!!')
    time_count(t0, time.time())
    
# Clean all stop words and save it to removed_stop_words.txt file        
#clean_all_stop_words(df, 'removed_stop_words.txt')

def word_fre(data_fitted, vocabulary):
    sum_of_word = np.sum(data_fitted,axis=0)
    
    for (word, count) in zip(vocabulary, sum_of_word):
        print(str(word) + ': ' + str(count))

def repare_data(data_frame, num_of_vocabulary):
    data_frame = data_frame[data_frame['Score'] != 3]; #ignore 3 stars reviews
    data_frame['sentiment'] = data_frame['Score'] >= 4
    train, test = train_test_split(data_frame, test_size=0.2)
    
    vectorizer = CountVectorizer(analyzer='word',max_features=num_of_vocabulary)
    train_text = train['Text'].values.astype('U')
    test_text = test['Text'].values.astype('U')
    
    X_train = vectorizer.fit_transform(train_text).toarray()
    Y_train = train['sentiment']
    
    X_test = vectorizer.fit_transform(test_text).toarray()
    Y_test = test['sentiment']
    
    vocabulary = vectorizer.get_feature_names()
    
    return X_train, Y_train, X_test, Y_test, vocabulary


def train_data(X_train, Y_train, X_test, Y_test):
    # list of classifier:
    t0= time.time()
    classifier_name = ['Decision Tree','Random Forest', 'Aadboost','Naive Bayes' ]
    classifier_model = [
        DecisionTreeClassifier(max_depth = 4),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB()]
    
    # Empty Dictionary
    result = {}
    # train data using scikit-learn lib
    for (name, model) in zip(classifier_name, classifier_model):
        score = model.fit(X_train, Y_train).score(X_test, Y_test)
        result[name] = score
    
    #Print the Results
    for name in result:
        print(name + ' : accurency ' + str(round(result[name], 4)))
    
    time_count(t0, time.time())
    
# RUN
if __name__ == '__main__':
    
    #1. PreProcesing data
    #Clean html tags on file
    #text_to_csv('test_raw.txt','test_removed_html.csv')
    text_to_csv('finefoods.txt','finefoods_removed_html.csv')
    print('\n')
    
    #Remove stop words
    #data_frame_hmtl = pd.read_csv('test_removed_html.csv')
    data_frame_hmtl = pd.read_csv('finefoods_removed_html.csv')
    #clean_all_stop_words(data_frame_hmtl, 'test_removed_stop_words.csv')
    clean_all_stop_words(data_frame_hmtl, 'finefoods_removed_stop_words.csv')
    
    #2. Train
    # Split data to train and test
    #data_frame = pd.read_csv('test_removed_stop_words.csv', nrows=10000)
    data_frame = pd.read_csv('finefoods_removed_stop_words.csv', nrows = 100000)
    num_of_vocabulary = 20 
    X_train, Y_train, X_test, Y_test, vocabulary = repare_data(data_frame, num_of_vocabulary)
    word_fre(X_train, vocabulary)
    print('\nTraining...')
    train_data(X_train, Y_train, X_test, Y_test)

    






    


    
    
    
    
    
    
    
    
    
    
    
    