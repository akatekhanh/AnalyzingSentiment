#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:58:15 2020

@author: akate/ Quoc Khanh UT
@purpuse: Analyzing sentiment using data "Amazone fine food data"
CLASSIFICATION

MODEL: 
    
ARGORITHM:
    
CONCEPT:
    
"""
# using dir(object) or help(object)

# Regular expression: how to use it
import urllib.request as urllib2
import re # reguler expression from lib
import nltk # natural language tookit
from bs4 import BeautifulSoup

response = urllib2.urlopen('https://www.nytimes.com/')
# get html code from url to process
html = response.read()
len_html = len(html)

# convert html string to token
tokens = [tok for tok in html.split()]

# another way is use re.split()
# convert byte type to string type
html_string = html.decode('ISO-8859-1')
tokens_re = re.split('\W+', html_string)
len_token_re = len(tokens_re)
#%%
# Remove html tag using nltk via clean_hmml() function
#token_nltk = nltk.clean_html(html_string)
from bs4 import BeautifulSoup
raw = BeautifulSoup(html).get_text()
print(raw[:100])
#%%
import operator
# using dictionary
freq_dis = {}
for tok in tokens_re:
    if tok in freq_dis:
        freq_dis[tok] += 1
    else:
        freq_dis[tok] = 1

sorted_freq_dis = sorted(freq_dis.items(), key = operator.itemgetter(1), reverse= True)
#%% Conver to csv file
name_to_csv =[]
name_to_count = []
for i in range(len(sorted_freq_dis) - 1):
    # convert from tupple to list and export to csv file
    name_to_csv.append(sorted_freq_dis[i][0])
    name_to_count.append(sorted_freq_dis[i][1])
# Export using pandas
import pandas as pd
# Data frame from list
data_frame = {'Element':name_to_csv, 'Count':name_to_count}
df = pd.DataFrame(data_frame, columns=['Element', 'Count'])
df.to_csv('WordPython.csv', header=True, index=False)































