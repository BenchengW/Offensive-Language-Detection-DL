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

transform = ShuffleSentencesTransform(p=2.0)

!pip install nlpaug

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action
aug_insert_bert = naw.ContextualWordEmbsAug(
      model_path='bert-base-uncased', action="insert")

aug_substitute_bert =  naw.ContextualWordEmbsAug(
    model_path='roberta-base', action="substitute")

aug_substitute_bert.aug_p=0.3

aug_wordnet = naw.SynonymAug(aug_src='wordnet')

aug_swap = naw.RandomWordAug(action="swap")

aug_delete = naw.RandomWordAug()


def data_augment_bert_sw(aug_insert, aug_substitute, aug_swap, text):
  
  augmented_text = aug_insert.augment(text)
  augmented_text = aug_substitute.augment(augmented_text)
  augmented_text = aug_swap.augment(augmented_text)

  # print("Original:")
  print(text)
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

def data_augment_Hate(df, class_number, number_of_aug):

    minority_df = df[df['class']==class_number].copy()
    minority_df = minority_df.reset_index(drop=True)
    
    minority_df_len = len(minority_df)

    final_df = pd.DataFrame()

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

    return final_df