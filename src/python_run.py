!git clone https://github.com/BenchengW/Hate_Speech_Detection_MMAI_894_DL
!pip install nlpaug
!pip install transformer

import os
import sys
os.chdir('Hate_Speech_Detection_MMAI_894_DL/src')

#################################################
# Load Albert pretrain model
################################################
from main import Albert_pretrain
Pretrain = Albert_pretrain()
Pretrain.load_albert()
Pretrain.predict("I happliy like it")
#Pretrain.predict(text_dataframe) or List of text

Pretrain.check_sentiment("I reallly hate this one")
Pretrain.doc_augmentation("I reallly hate this one")
#Pretrain.corpus_augmentation(text_dataframe)
#Pretrain


#################################################
# Load Albert model and train on your own data
################################################
from main import *
New_data = load_data()
Albert_model = Albert(New_data, 50, 2)    #50 batch size and 2 epoch
Albert_model.fit_albert()
Albert_model.predict("I happliy like it")
#Albert_model.predict(text_dataframe) or List of text

Albert_model.check_sentiment("I reallly hate this one")
Albert_model.doc_augmentation("I reallly hate this one")
#Albert_model.corpus_augmentation(text_dataframe)