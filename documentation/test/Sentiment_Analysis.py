# -*- coding: utf-8 -*-
"""Hate Speech Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1x6xXTLOWnUsmDqFn2HEXwiBgaokMb6Vj
"""

# Clone Github Repo which includes sample data
!git clone https://github.com/BenchengW/Hate_Speech_Detection_MMAI_894_DL

#!pip install transformers
#!pip install pandas

import matplotlib.pyplot as plt
import tensorflow.keras as keras
import pandas as pd

#try: # this is only working on the 2nd try in colab :)
#  from transformers import DistilBertTokenizer, TFDistilBertModel
#except Exception as err: # so we catch the error and import it again
#  from transformers import DistilBertTokenizer, TFDistilBertModel

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalMaxPool1D, Flatten
from pandas_profiling import ProfileReport
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
import tensorflow as tf

#dbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Commented out IPython magic to ensure Python compatibility.
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
# %matplotlib inline

Hate_speech = pd.read_csv("/content/Hate_Speech_Detection_MMAI_894_DL/data/labeled_data.csv")
Hate_speech.head(5)



#Code to remove @
Hate_speech['clean_tweet'] = Hate_speech['tweet'].apply(lambda x : ' '.join([tweet for tweet in x.split()if not tweet.startswith("@")]))

# remove special characters, numbers, punctuations
Hate_speech['tidy_tweet'] = Hate_speech['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

# remove all the words having length 3 or less. For example, terms like “hmm”, “oh” are of very little use. It is better to get rid of them.
Hate_speech['tidy_tweet2'] = Hate_speech['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
Hate_speech.head(3)

tokenized_tweet = Hate_speech['tidy_tweet2'].apply(lambda x: x.split())
tokenized_tweet.head()

# from nltk.stem.porter import *
# stemmer = PorterStemmer()

# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
# tokenized_tweet.head()

# for i in range(len(tokenized_tweet)):
    # tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

# Hate_speech['tidy_tweet_af_stem'] = tokenized_tweet
# Hate_speech.head(5)

Hate_speech['tidy_tweet2'][2]

from textblob import TextBlob
def get_sentiment_subjectivity_score(text):
  analysis = TextBlob(text)
  return analysis.sentiment[1]

def get_sentiment_polarity_score(text):
  analysis = TextBlob(text)
  return analysis.sentiment[0]

def get_sentiment_polarity(corpus):
    res = np.array([get_sentiment_polarity_score(instance) for instance in corpus]).reshape(-1, 1)
    return res

def get_sentiment_subjectivity(corpus):
    res = np.array([get_sentiment_subjectivity_score(instance) for instance in corpus]).reshape(-1, 1)
    return res

Hate_speech['subjectivity'] = get_sentiment_subjectivity(Hate_speech['tidy_tweet2'])

Hate_speech['polarity'] = get_sentiment_polarity(Hate_speech['tidy_tweet2'])

def get_name(x):
    if x == 0:
        return "Hate"
    elif x ==1:
        return "Offensive"
    else:
        return 'Neither'

Hate_speech['class_name'] = Hate_speech['class'].apply(get_name)

Hate_speech

Hate_speech.groupby(by = "class_name")['polarity'].mean().plot(kind = "bar", title="Sentiment - polarity")

Hate_speech.groupby(by = "class_name")['subjectivity'].mean().plot(kind = "bar", title="Sentiment - subjectivity")

analysis = TextBlob(Hate_speech['tidy_tweet2'][2])
analysis.sentiment



# Word Cloud
all_words_class1 = ' '.join([text for text in Hate_speech[Hate_speech['class']==1]['tidy_tweet_af_stem']])
from wordcloud import WordCloud
wordcloud_class1 = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_class1)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_class1, interpolation="bilinear")
plt.axis('off')
plt.show()

all_words_class2 = ' '.join([text for text in Hate_speech[Hate_speech['class']==2]['tidy_tweet_af_stem']])
from wordcloud import WordCloud
wordcloud_class2 = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_class2)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_class2, interpolation="bilinear")
plt.axis('off')
plt.show()

all_words_class0 = ' '.join([text for text in Hate_speech[Hate_speech['class']==0]['tidy_tweet_af_stem']])
from wordcloud import WordCloud
wordcloud_class0 = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words_class0)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_class0, interpolation="bilinear")
plt.axis('off')
plt.show()

# Understanding the impact of Hashtags on tweets sentiment

# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from nethier tweets
HT_nether = hashtag_extract(Hate_speech['tidy_tweet_af_stem'][Hate_speech['class'] == 2])

# extracting hashtags from hate tweets
HT_hate = hashtag_extract(Hate_speech['tidy_tweet_af_stem'][Hate_speech['class'] == 0])

# extracting hashtags from offensive tweets
HT_offensive = hashtag_extract(Hate_speech['tidy_tweet_af_stem'][Hate_speech['class'] == 1])

# unnesting list
HT_nether = sum(HT_nether,[])
HT_hate = sum(HT_hate,[])
HT_offensive = sum(HT_offensive,[])

a = nltk.FreqDist(HT_nether)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Neither Hashtag tweets")
plt.show()

a = nltk.FreqDist(HT_offensive)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hate offensive tweets")
plt.show()

a = nltk.FreqDist(HT_hate)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.title("Hate Hashtag tweets")
plt.show()



























