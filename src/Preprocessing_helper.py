import transformers
import tensorflow.keras as keras
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import zeros
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, Bidirectional, LSTM, GRU, Flatten, LayerNormalization, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

from nltk.tokenize import TweetTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, TFDistilBertModel
from transformers import BertTokenizer, TFBertModel, TFGPT2Model, GPT2Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import seaborn as sn
import matplotlib.pyplot as plt

import re
import tensorflow as tf

###############################################################################################################
#these are preprocess function
#################################################################################################################
#load data
def load_data():
  raw_data_df = pd.read_csv('https://query.data.world/s/twuhmzuhvitwqqcjh5picrq3qykr4r')
  return raw_data_df

def hashtag(text):
  FLAGS = re.MULTILINE | re.DOTALL
  text = text.group()
  hashtag_body = text[1:]
  if hashtag_body.isupper():
      result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
  else:
      result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
  return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> " 

def clean_data(text):
  FLAGS = re.MULTILINE | re.DOTALL
  eyes = r"[8:=;]"
  nose = r"['`\-]?"
  def re_sub(pattern, repl):
      return re.sub(pattern, repl, text, flags=FLAGS)

  text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
  text = re_sub(r"@\w+", "<user>")
  text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
  text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
  text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
  text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
  text = re_sub(r"/"," / ")
  text = re_sub(r"<3","<heart>")
  text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
  text = re_sub(r"#\w+", hashtag)  
  text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
  text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

  text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
  text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
  text = re_sub(r"  ", r" ")
  text = re_sub(r" ([A-Z]){2,} ", allcaps)
    
  return text.lower()

def preprocessing_tweet(tweet_df):
  temp_list= []
  for t in tweet_df['tweet']:
    temp_list.append(clean_data(t))
  tweet_df['clean_tweet'] = temp_list
  return tweet_df
###########################################################################################################

def Albert_Sentiment(text):

  analysis = TextBlob(text)
  print("#"*100)
  print("#")
  print("#Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement.\n#Subjective sentences generally refer to personal opinion, emotion or judgment also range of [0,1].")
  print("#")
  print("#"*100)
  print("\nPolarity is {}".format(analysis.sentiment[0]))
  print("Subjective is {}".format(analysis.sentiment[1]))

