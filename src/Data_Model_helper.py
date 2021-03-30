###############################################################################################################
# These are model helper
#################################################################################################################

#split train val test
def split_data(cleantweet):
  trainX, tempX = train_test_split(cleantweet, test_size=0.4, random_state=42)
  valX, testX = train_test_split(tempX, test_size=0.5, random_state=42)
  return trainX, valX, testX

#extract tweet and y
def extract_tweet_and_y(raw_data_df):
  tweet, target = raw_data_df['clean_tweet'], raw_data_df['class']
  return tweet, target

#tokenize and vectorize input using keras tokenizer
def keras_tokenizer(tweet_train, tweet_val, tweet_test, maxnumwords):
  # maxnumwords = 2000
  kt = Tokenizer()
  kt.fit_on_texts(tweet_train)
  word_index = kt.word_index
  vocab_size = len(word_index) + 1

  train_vectors = kt.texts_to_sequences(tweet_train) #Converting text to a vector of word indexes
  val_vectors = kt.texts_to_sequences(tweet_val) #Converting text to a vector of word indexes
  test_vectors = kt.texts_to_sequences(tweet_test) #Converting text to a vector of word indexes
  
  train_padded = pad_sequences(train_vectors, maxlen=maxnumwords, padding='post')
  val_padded = pad_sequences(val_vectors, maxlen=maxnumwords, padding='post')
  test_padded = pad_sequences(test_vectors, maxlen=maxnumwords, padding='post')

  return  train_padded, val_padded, test_padded, vocab_size, word_index


#tokenize and vectorize input using keras tokenizer
def tweet_tokenizer(tweet, maxnumwords):
  # maxnumwords = 2000
  kt = Tokenizer()
  kt.fit_on_texts(tweet)

  tweet_vectors = kt.texts_to_sequences(tweet) #Converting text to a vector of word indexes
  tweet_padded = pad_sequences(tweet_vectors, maxlen=maxnumwords, padding='post')

  return tweet_padded

#GloVe embeddings using Glove twiter 100D
def GloveTwitterEmbedding(vocab_size, word_index):
  
  #Glove Twitter 100d
  embedding_path = "/content/drive/MyDrive/Colab Notebooks/GloVe Twitter 27B/glove.twitter.27B.100d.txt"
  # max_features = 30000
  # get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
  # embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
  embedding_index = dict(
      (o.strip().split(" ")[0], np.array(o.strip().split(" ")[1:], dtype="float32")
      ) for o in open(embedding_path)
      )
  # embedding matrix
  embedding_matrix = zeros((vocab_size, 100))
  # for word, i in enumerate(tweet_tokenized):
  for t , i in enumerate(word_index.items()):
    embedding_vector = embedding_index.get(t)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  
  return embedding_matrix

# HuggingFace Transformers AutoTokenizer
def hf_auto_tokenizer(tweet, maxnumwords):
  
  autoTokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  tweet_tokenized = autoTokenizer(tweet.tolist(), padding = 'max_length',
                                  truncation = True, max_length = maxnumwords, return_tensors='tf')
  input_ids = tweet_tokenized['input_ids']
  att_mask = tweet_tokenized['attention_mask']
  return input_ids, att_mask

#HuggingFace GPT2 Tokenizer
def hf_GPT2_tokenizer(tweet, maxnumwords):

  gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  gpt2Tokenizer.pad_token = gpt2Tokenizer.eos_token
  tweet_tokenized = gpt2Tokenizer(tweet.tolist(), padding = 'max_length', truncation = True, max_length = maxnumwords, return_tensors='tf')
  input_ids = tweet_tokenized['input_ids']
  att_mask = tweet_tokenized['attention_mask']
  return input_ids, att_mask

#one hot encode y
def prepare_target(raw_y):
  class_weight = compute_class_weight('balanced', np.arange(3), raw_y)
  class_weight_dict = dict((c,w) for c, w in enumerate(class_weight))
  target = to_categorical(raw_y)
  return np.array(target), class_weight_dict
  
  
  
  
#########################################################
# model list
##########################################################
def albert_model(param={}):
  #Bi Directional LSTM
  max_seq_len = param['Max_length']
  inputs = Input(shape = (max_seq_len,), dtype='int64', name='inputs')

  vocab_size = param['Vocab Size']

  embedding_trainable = True
  e = Embedding(vocab_size, 100, embeddings_initializer ='uniform', 
                input_length=max_seq_len, trainable = embedding_trainable)
  
  embedding_matrix = param['Embedding Matrix']
  if embedding_matrix is not None:
    embedding_trainable = False
    e = Embedding(vocab_size, 100, embeddings_initializer ='uniform', input_length=max_seq_len,
                      weights = [embedding_matrix], trainable = embedding_trainable)

  model = Sequential()
  model.add(inputs)
  model.add(e)
  model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=param['dropout']), merge_mode='concat'))
  model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=param['dropout']),merge_mode='concat'))
  model.add(Flatten())
  model.add(LayerNormalization())
  model.add(Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(param['dropout']))
  model.add(LayerNormalization())
  model.add(Dense(param['second_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(param['dropout']))
  model.add(Dense(3, activation='softmax'))

  model.summary()
  return model

def tl_disbert_model(param={}):
  
  trainable = param['Trainable']
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  disBert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
  disBert_model.trainable = param['Trainable']

  disBert_output = disBert_model(inputs, attention_mask = masks)
  disBert_last_hidden = disBert_output.last_hidden_state
  disBert_CLS_output =  disBert_last_hidden [:,0,:]
  x = Flatten()(disBert_CLS_output)
  x = LayerNormalization()(x)
  x = Dense(param['first_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()

  return model

def tl_bert_model(param={}):

  
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  Bert_model = TFBertModel.from_pretrained('bert-base-uncased')
  Bert_model.trainable = param['Trainable']

  Bert_output = Bert_model(inputs, attention_mask = masks)
  Bert_last_hidden = Bert_output.last_hidden_state
  Bert_CLS_output =  Bert_last_hidden [:,0,:]
  x = LayerNormalization()(Bert_CLS_output)
  x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()
  return model

def tl_gpt2_model(param={}):
  
  trainable = param['Trainable']
  max_seq_len = param['Max_length']
  inputs = Input(shape= (max_seq_len,), dtype ='int64', name='inputs')
  masks = Input(shape = (max_seq_len,), dtype='int64', name='masks')

  gpt2_model = TFGPT2Model.from_pretrained('gpt2')
  gpt2_model.trainable = param['Trainable']

  gpt2_output = gpt2_model(inputs, attention_mask = masks)
  gpt2_last_hidden = gpt2_output.last_hidden_state
  # gpt2_CLS_output =  gpt2_last_hidden[:,0,:]
  x = Flatten()(gpt2_last_hidden)
  x = LayerNormalization()(x)
  x = Dense(param['first_layer'], activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Dropout(param['dropout'])(x)
  x = LayerNormalization()(x)
  x = Dense(param['second_layer'], activation='relu')(x)
  x = Dropout(param['dropout'])(x)

  probs = Dense(3, activation='softmax')(x)

  model = keras.Model(inputs = [inputs, masks], outputs=probs)
  model.summary()
  return model
  
  
  
###################################################
# Model train helper
##################################################


def train_model(model, tweet_train, y_train, tweet_val, y_val, batch_size, num_epochs, class_weight):
  
  es = keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=3)
  history = model.fit(
            tweet_train, y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,
            validation_data=(tweet_val, y_val),
            class_weight = class_weight,
            callbacks=[es])
  return model, history

def compile_model(model):
  model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics=['accuracy', 
                         keras.metrics.AUC(curve="ROC", multi_label=True), 
                         keras.metrics.AUC(curve="PR", multi_label=True), 
                         keras.metrics.Precision(),
                         keras.metrics.Recall()])
  return model

#Create Batch Prediction for out of GPU memory solution
def model_batch_predict(model, model_inputs_and_masks_test, batch_size=100):
    
    probs = np.empty((0,3))
    iteration = int(model_inputs_and_masks_test['inputs'].shape[0] / batch_size)
    # last_batch = model_inputs_and_masks_test.shape[1] % batch_size
    i = 0
    if type(model_inputs_and_masks_test) is dict:
      for i in range(iteration):
        test = {'inputs':model_inputs_and_masks_test['inputs'][i*batch_size:(i+1)*batch_size], 
                'masks': model_inputs_and_masks_test['masks'][i*batch_size:(i+1)*batch_size]}
        probs= np.concatenate((probs, np.array(model(test, training=False))))
      last_batch_test =  {'inputs':model_inputs_and_masks_test['inputs'][(i+1)*batch_size:],
                          'masks': model_inputs_and_masks_test['masks'][(i+1)*batch_size:]}
      probs= np.concatenate((probs, np.array(model(last_batch_test, training=False))))

    else:
      probs = model(model_inputs_and_masks_test, training=False)

    return np.array(probs)


#Create Batch Prediction for out of GPU memory solution
def model_predict(model, model_inputs_and_masks_test, batch_size=100):
    
    probs = model(model_inputs_and_masks_test, training=False)

    return np.array(probs)


#Create Batch Prediction for out of GPU memory solution
def model_batch_predict(model, model_inputs_and_masks_test):
    

      test = {'inputs':model_inputs_and_masks_test['inputs'][i*batch_size:(i+1)*batch_size], 
                'masks': model_inputs_and_masks_test['masks'][i*batch_size:(i+1)*batch_size]}
        probs= np.concatenate((probs, np.array(model(test, training=False))))
      last_batch_test =  {'inputs':model_inputs_and_masks_test['inputs'][(i+1)*batch_size:],
                          'masks': model_inputs_and_masks_test['masks'][(i+1)*batch_size:]}
      probs= np.concatenate((probs, np.array(model(last_batch_test, training=False))))

    else:
      probs = model(model_inputs_and_masks_test, training=False)

    return np.array(probs)


def evaluate_model(probs, y_test):
    # print(probs)
    # print(y_test)

    eval_dict = {
        "Hate": {
            "pr_auc": average_precision_score(y_test[:, 0], probs[:, 0]), "pr_auc_random_guess": sum(y_test[:, 0])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 0], probs[:, 0]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 0], probs[:, 0] > 0.2),
            "recall": recall_score(y_test[:, 0], probs[:, 0] > 0.2)
        }, 
        "Offensive": {
            "pr_auc": average_precision_score(y_test[:, 1], probs[:, 1]), "pr_auc_random_guess": sum(y_test[:, 1])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 1], probs[:, 1]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 1], probs[:, 1] > 0.2),
            "recall": recall_score(y_test[:, 1], probs[:, 1] > 0.2)
        }, 
        "Neither": {
            "pr_auc": average_precision_score(y_test[:, 2], probs[:, 2]), "pr_auc_random_guess": sum(y_test[:, 2])/(1.0*y_test.shape[0]), 
            "roc_auc": roc_auc_score(y_test[:, 2], probs[:,2]), "roc_auc_random_guess": 0.5, 
            "precision": precision_score(y_test[:, 2], probs[:, 2] > 0.2),
            "recall": recall_score(y_test[:, 2], probs[:, 2] > 0.2)
        }
    }
    return eval_dict


def plot_confusion_matrix(predict, y_true):
  y_predict = predict.argmax(1)
  class_hate = pd.DataFrame(confusion_matrix(y_test[:,0], y_predict==0))
  class_offensive = pd.DataFrame(confusion_matrix(y_test[:,1], y_predict==1))
  class_neither = pd.DataFrame(confusion_matrix(y_test[:,2], y_predict==2))

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
  sn.set(font_scale=1.5)#for label size
  sn.heatmap(class_hate, cmap="cool", annot=True, fmt='g', ax=ax1, cbar=False)
  sn.heatmap(class_offensive, cmap="Greens", annot=True, fmt='g', ax=ax2, cbar=False)
  sn.heatmap(class_neither, cmap="YlGnBu", annot=True, fmt='g', ax=ax3, cbar=False)

  ax1.set_ylabel('True')
  ax2.set_ylabel('True')
  ax3.set_ylabel('True')
  ax1.set_xlabel('Predicted')
  ax2.set_xlabel('Predicted')
  ax3.set_xlabel('Predicted')
  ax1.set_title('Hate')
  ax2.set_title('Offensive')
  ax3.set_title('Neither')

  plt.tight_layout()
  plt.show

def print_classification_report(predict, y_true):
  y_predict = predict.argmax(1)
  class_hate = classification_report(y_test[:,0], y_predict==0)
  class_offensive = classification_report(y_test[:,1], y_predict==1)
  class_neither = classification_report(y_test[:,2], y_predict==2)

  print("Hate Speech".center(60), "\n", class_hate, "\n\n", 
        "Offensive Speech".center(60), '\n', class_offensive, '\n', 
        "Neither".center(60), '\n', class_neither)
