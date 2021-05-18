#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import re
import string
import time
from nltk.tokenize import sent_tokenize, RegexpTokenizer


# In[2]:


cik_list = pd.read_csv('cik_list.csv')


# In[3]:


#fetching data
def extract_data(link):
    link = 'https://www.sec.gov/Archives/' + link.strip()
    headers = {"user-agent": "Mozilla/5.0"}
    f = requests.get(link,headers=headers)
    text = f.text
    return text


# In[4]:


report = cik_list['SECFNAME'].apply(extract_data)


# In[5]:


#cleaning data
def clean_data(text):
    #Remove HTML Tags
    text = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});','', text)
    
    #remove extra line and tabs
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')

    #remove punctuation 
#     text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers and special characters
    text = re.sub(r'[^a-zA-z.,!?/:;\"\'\s]',' ',text)
    
    #remove multiple spaces
    text = re.sub('(?s) +',' ',text)
    return text


# In[6]:


report_data = {'raw_text':report}


# In[7]:


report_df = pd.DataFrame(report_data)


# In[8]:


report_df['clean_data'] = report_df['raw_text'].apply(clean_data)


# # StopWords

# In[9]:


f = open("StopWords_Generic.txt", "r")
stop_words = f.read().lower()


# In[10]:


stopWordList = stop_words.split('\n')


# In[11]:


def tokenize_text(text):
    tokenizer = RegexpTokenizer(r'\w+') #removing punctuation
    tokens = tokenizer.tokenize(text.lower())
    filtered_words = list(filter(lambda token: token not in stopWordList, tokens)) # filtering stopwords
    return filtered_words


# In[12]:


report_df['filtered'] = report_df['clean_data'].apply(tokenize_text)


# In[13]:


positive_df = pd.read_csv('Positive-Table.csv')
negative_df = pd.read_csv('Negative-Table.csv')


# In[14]:


positiveWords = positive_df['Unnamed: 0'].apply(lambda x:x.lower())


# In[15]:


positiveWordsList = positiveWords.tolist()


# In[16]:


negativeWords = negative_df['Unnamed: 0'].apply(lambda x:x.lower())


# In[17]:


negativeWordsList = negativeWords.tolist()


# In[18]:


positiveWordsList = list(filter(lambda word: word not in stopWordList, positiveWordsList))


# In[19]:


negativeWordsList = list(filter(lambda word: word not in stopWordList, negativeWordsList))


# # Calculating scores

# In[20]:


# Calculating positive score 
def positive_score(token):
    posWords = 0
    for word in token:
        if word in positiveWordsList:
            posWords  += 1
    return posWords


# In[21]:


# Calculating Negative score
def negative_score(token):
    negWords=0
    for word in token:
        if word in negativeWordsList:
            negWords -=1
    return negWords*-1


# In[22]:


# Calculating polarity score
def polarity_score(positiveScore, negativeScore):
    pol_score = (positiveScore - negativeScore) / ((positiveScore + negativeScore) + 0.000001)
    return pol_score


# In[23]:


report_df['positive_score'] = report_df['filtered'].apply(positive_score)


# In[24]:


report_df['negative_score'] = report_df['filtered'].apply(negative_score)


# In[25]:


report_df['polarity_score'] = report_df.apply(lambda x: polarity_score(x.positive_score,x.negative_score),axis=1)


# In[26]:


#avg_sentence_length
def average_sentence_length(text,word_token):
    sentence_token = sent_tokenize(text)
    totalWordCount = len(word_token)
    totalSentences = len(sentence_token)
    average_sent_length = 0
    if totalSentences != 0:
        average_sent_length = totalWordCount / totalSentences    
    return round(average_sent_length)


# In[27]:


report_df['average_sentence_length'] = report_df.apply(lambda x: average_sentence_length(x.clean_data,x.filtered),axis=1)


# In[28]:


#syllable count
def syllable_count(word):
    vowels = 0
    word = word.lower()
    if word.endswith(('es','ed')):
            pass
    else:
        for w in word:
            if(w=='a' or w=='e' or w=='i' or w=='o' or w=='u'):
                vowels += 1
    return vowels


# In[29]:


#complex words
def complex_word_count(token):
    complexWords = 0
    for word in token:
        if syllable_count(word) > 2:
            complexWords+=1
    return complexWords


# In[30]:


#calculating complex word percentage
def complex_word_percentage(token):
    totalWords = len(token)
    complexWords = complex_word_count(token)
    return complexWords/totalWords


# In[31]:


report_df['percentage_of_complex_words'] = report_df['filtered'].apply(complex_word_percentage)


# In[32]:


#fog index
def fog_index(avg_sentence_length,percentage_complex):
    return 0.4*(avg_sentence_length+percentage_complex)


# In[33]:


report_df['fog_index'] = report_df.apply(lambda x:fog_index(x.average_sentence_length,x.percentage_of_complex_words),axis=1)


# In[34]:


report_df['word_count'] = report_df['filtered'].apply(lambda x:len(x))


# In[35]:


report_df['complex_word_count'] = report_df['filtered'].apply(complex_word_count)


# In[36]:


uncertainty_df = pd.read_csv('uncertainty_dictionary.csv')


# In[37]:


uncertainWords = uncertainty_df['Word'].apply(lambda x:x.lower())
uncertainWordsList = uncertainWords.tolist()


# In[38]:


# Calculating uncertainity score 
def uncertainty_score(token):
    uncWords = 0
    for word in token:
        if word in uncertainWordsList:
            uncWords  += 1
    return uncWords


# In[39]:


constraining_df = pd.read_csv('constraining_dictionary.csv')
constrainWords = constraining_df['Word'].apply(lambda x:x.lower())
constrainWordsList = constrainWords.tolist()


# In[40]:


# Calculating constraining score 
def constraining_score(token):
    constrainWords = 0
    for word in token:
        if word in constrainWordsList:
            constrainWords  += 1
    return constrainWords


# In[41]:


report_df['uncertainty_score'] = report_df['filtered'].apply(uncertainty_score)


# In[42]:


report_df['constraining_score'] = report_df['filtered'].apply(constraining_score)


# In[43]:


# Positive word proportion
def positive_word_proportion(positiveScore,wordcount):
    pwp = 0
    if wordcount !=0:
        pwp = positiveScore / wordcount
    return pwp


# In[44]:


# Negative word proportion
def negative_word_proportion(negativeScore,wordcount):
    nwp = 0
    if wordcount !=0:
        nwp = negativeScore / wordcount
    return nwp


# In[45]:


report_df['positive_word_proportion'] = report_df.apply(lambda x:positive_word_proportion(x.positive_score,x.word_count),axis=1)


# In[46]:


report_df['negative_word_proportion'] = report_df.apply(lambda x:negative_word_proportion(x.negative_score,x.word_count),axis=1)


# In[47]:


# Uncertain word proportion
def uncertain_word_proportion(uncertainScore,wordcount):
    uwp = 0
    if wordcount !=0:
        uwp = uncertainScore / wordcount
    return uwp


# In[48]:


# Constraining word proportion
def constrain_word_proportion(constrainScore,wordcount):
    cwp = 0
    if wordcount !=0:
        cwp = constrainScore / wordcount
    return cwp


# In[49]:


report_df['uncertainty_word_proportion'] = report_df.apply(lambda x:uncertain_word_proportion(x.uncertainty_score,x.word_count),axis=1)


# In[50]:


report_df['constraining_word_proportion'] = report_df.apply(lambda x:constrain_word_proportion(x.constraining_score,x.word_count),axis=1)


# In[51]:


report_df['constraining_words_whole_report'] = report_df['filtered'].apply(constraining_score)


# In[52]:


final_report = cik_list.join(report_df.iloc[:,3:])


# In[53]:


final_report.to_csv('final_report.csv')

