#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler


# In[2]:


EMBEDDING_FILES = [
    '/home/ravi/embeds/crawl-300d-2M.vec',
    '/home/ravi/embeds/glove.840B.300d.txt'
]


# In[3]:


NUM_MODELS = 2
BATCH_SIZE = 512
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 4
MAX_LEN = 220


# In[4]:


IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = 'target'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'


# In[5]:


from typing import List


# In[6]:


# a string of literals to work and embeds pair
def get_pairs(word,*embeds):
    return word,np.array(embeds)


# In[7]:


get_pairs('dsa','1','2','3','5','6')


# In[8]:


f_name = EMBEDDING_FILES[0]


# In[9]:


get_ipython().system('ls')


# In[10]:


f = open('cat.txt')


# In[11]:


a = f.readlines(100)


# In[12]:


a


# In[13]:


#iterate through the lines in a file:
# make this step faster to make pipeline faster
# insted of line in file made it to line in readlines(char)
embed_dict = {}
with open(f_name) as f:
        embed_dict = dict([get_pairs( *line.strip().split(' ') ) for line in f.readlines(7000)])


# In[14]:


def make_line(word,*embed):
    return word,np.array(embed)


# In[15]:


import gc


# In[16]:


del embed_dict
gc.collect()


# In[17]:


make_line(1,2,3)


# In[18]:


dict(((1,2),(3,4)))


# In[19]:


embeds = {}


# In[20]:


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

# edited to make pipeline faster
def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f.readlines(5000))


def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
    

def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
    


# In[21]:


train_df = pd.read_csv('../../data/train_min.csv')
test_df = pd.read_csv('../../data/test_min.csv')


# In[22]:


train_df.shape,test_df.shape


# In[23]:


x_train = train_df[TEXT_COLUMN].astype(str)
y_train = train_df[TARGET_COLUMN].values
y_aux_train = train_df[AUX_COLUMNS].values
x_test = test_df[TEXT_COLUMN].astype(str)


# In[24]:


#train_df.iloc[:3]


# In[25]:


#train_df_min = train_df.iloc[:1000]
#test_df_min = test_df.iloc[:1000]


# In[26]:


#test_df_min.to_csv('../../data/test_min.csv',index=False)
#train_df_min.to_csv('../../data/train_min.csv',index=False)


# In[27]:


#_ = pd.read_csv('../../data/train_min.csv')


# In[28]:


#_.head(2)


# In[29]:


#train_df.head(2)


# In[30]:


#train_df[IDENTITY_COLUMNS].describe()


# In[31]:


for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:
    train_df[column] = np.where(train_df[column] >= 0.5, True, False)


# In[32]:


tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_train) + list(x_test))


# In[33]:


x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# In[34]:


sample_weights = np.ones(len(x_train), dtype=np.float32)


# In[35]:


sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)


# In[36]:


sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)


# In[37]:


sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5


# In[38]:


sample_weights /= sample_weights.mean()


# In[39]:


embedding_matrix = np.concatenate(
    [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)


# In[40]:


checkpoint_predictions = []
weights = []


# In[41]:


from tqdm import tqdm_notebook as tqdm


# In[42]:


from tensorflow.contrib.rnn import *


# In[43]:


import tensorflow as tf
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)


# In[44]:


with tf.Session() as sess:
    print (sess.run(c))


# In[45]:


for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1])
    for global_epoch in range(EPOCHS):
        model.fit(
            x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=2,
            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],
            callbacks=[
                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))
            ]
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
        weights.append(2 ** global_epoch)


# In[ ]:


predictions = np.average(checkpoint_predictions, weights=weights, axis=0)


# In[ ]:


submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




