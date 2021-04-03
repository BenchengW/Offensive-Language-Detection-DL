# Hate_Speech_Detection_MMAI_894
 This project is going to leverage Deep Learing Technique such as NLP, Embedings, Word2Vec, Doc2Vec, Transfer Learning etc to detect hate speech.

### Summary of the Dataset: Hate Speech Data
Contributors viewed short text and identified if it a) contained hate speech, b) was offensive but without hate speech, or c) was not offensive at all. Contains nearly 15K rows with three contributor judgments per text string.
Link:https://toolbox.google.com/datasetsearch/search?query=labelled%20speech%20data&docid=DL25D4pKu50vjlI2AAAAAA%3D%3D 


### Instructions Notebook to run in Google colab or local machine
Link: https://github.com/BenchengW/Hate_Speech_Detection_MMAI_894_DL/blob/main/documentation/Instructions.ipynb

### Clone the project and start to run in colab:

clone the repo and set the repo as working directory

```bash
!git clone https://github.com/BenchengW/Hate_Speech_Detection_MMAI_894_DL
!pip install nlpaug
!pip install transformers
!pip install textblob
import os
import sys
os.chdir('Hate_Speech_Detection_MMAI_894_DL/src')
```

## Instructions to run

To run Albert Pretrain Model
BI-LSTM based models for twitter Hate speech detection

```
from main import Albert_pretrain
Pretrain = Albert_pretrain()
Pretrain.load_albert()

#Tweet Classificaiton
prediction = Pretrain.predict("I hate you a lot")
#Pretrain.predict(text_dataframe) or List of text

#Check Sentiment of Text
Pretrain.check_sentiment("I reallly hate this one

#Data Augmentation of Text
Pretrain.doc_augmentation("I reallly hate this one")
#Pretrain.corpus_augmentation(text_dataframe)
```

To run Albert Model and train on your own data
BI-LSTM based models for twitter Hate speech detection

```
from main import *
Albert_model = Albert(load_data(), batch_size, epoch)
Albert_model.fit_albert()

#Tweet Classificaiton
Albert_model.predict("I hate you a lot")
#Pretrain.predict(text_dataframe) or List of text

#Check Sentiment of Text
Albert_model.check_sentiment("I reallly hate this one

#Data Augmentation of Text
Albert_model.doc_augmentation("I reallly hate this one")
#Albert_model.corpus_augmentation(text_dataframe)
```