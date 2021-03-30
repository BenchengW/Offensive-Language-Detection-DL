################################################################################
# data Augmentation
################################################################################
import random
import re
import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm
from albumentations.core.transforms_interface import DualTransform, BasicTransform
import nltk
nltk.download('punkt')

SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
import TextBlob

from nlpaug.util import Action
aug_insert_bert = naw.ContextualWordEmbsAug(
      model_path='bert-base-uncased', action="insert")

aug_substitute_bert =  naw.ContextualWordEmbsAug(
    model_path='roberta-base', action="substitute")

aug_substitute_bert.aug_p=0.3

aug_wordnet = naw.SynonymAug(aug_src='wordnet')

aug_swap = naw.RandomWordAug(action="swap")

aug_delete = naw.RandomWordAug()

class NLPTransform(BasicTransform):
    """ Transform for nlp task."""
    LANGS = {
        'en': 'english',
        'it': 'italian', 
        'fr': 'french', 
        'es': 'spanish',
        'tr': 'turkish', 
        'ru': 'russian',
        'pt': 'portuguese'
    }

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(text, self.LANGS.get(lang, 'english'))

class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)
        return ' '.join(sentences), lang


def data_augment_bert_sw(aug_insert, aug_substitute, aug_swap, text):
  
  augmented_text = aug_insert.augment(text)
  augmented_text = aug_substitute.augment(augmented_text)
  augmented_text = aug_swap.augment(augmented_text)

  # print("Original:")
  # print(text)
  # print("Augmented Text:")
  # print(augmented_text)
  
  return augmented_text

def data_augment_wordnet_de(aug_insert, aug_wordnet,aug_delete, text):
  
  augmented_text = aug_insert.augment(text)
  augmented_text = aug_wordnet.augment(augmented_text)
  augmented_text = aug_delete.augment(augmented_text)
  # print("Original:")
  # print(text)
  # print("Augmented Text:")
  # print(augmented_text)
  
  return augmented_text


def Sentence_order_switch(text):
  lang = 'en'
  # print("Original Text")
  # print(text)
  # print("Redorder Text")
  Redorder_txt = transform(data=(text, lang))['data'][0]

  return Redorder_txts

def docs_augment(df, class_number, number_of_aug):

    print("Tips:Please make sure you have a column called 'clean_tweet'")
    final_df = pd.DataFrame()
    try:
      minority_df = df[df['class']==class_number].copy()
      minority_df = minority_df.reset_index(drop=True)
      
      minority_df_len = len(minority_df)

      while number_of_aug >0:

        if number_of_aug-minority_df_len>=0:

          new_df = minority_df.copy()
          new_df['tweet_agument'] = new_df.apply(lambda row : data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, row['clean_tweet']), axis =1)
          final_df = pd.concat([final_df, new_df])

          number_of_aug = number_of_aug - minority_df_len

        else:
          
          new_df = minority_df.iloc[0:number_of_aug].copy()
          new_df['tweet_agument'] = new_df.apply(lambda row : data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, row['clean_tweet']), axis =1)
          final_df = pd.concat([final_df, new_df])

          number_of_aug = 0

      del final_df['clean_tweet']

      final_df.rename(columns={'tweet_agument':'clean_tweet'}, inplace=True)
    except:
        print("Error: please try different format of input")

    return final_df


def doc_augment(text):
    doc_aug = data_augment_bert_sw(aug_insert_bert, aug_substitute_bert, aug_swap, text)

    print("Original Text:")
    print(text)
    print("Augmented Text:")
    doc_aug

    return doc_aug
    
    
    
    
###############################################################################
# Sentiment Analysis
##############################################################################

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
    
    
def Albert_Sentiment(Text):

  analysis = TextBlob(Text)
  print("#"*100)
  print("#")
  print("#Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement.\n#Subjective sentences generally refer to personal opinion, emotion or judgment also range of [0,1].")
  print("#")
  print("#"*100)
  print("\nPolarity is {}".format(analysis.sentiment[0]))
  print("Subjective is {}".format(analysis.sentiment[1]))