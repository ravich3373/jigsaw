#In[]:
import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from model_utils import *
from perprocess_utils import *
from train_utils import *
# In[1]:


CRAWL_EMBEDDING_PATH = '/home/ravi/embeds/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '/home/ravi/embeds/glove.840B.300d.txt'
NUM_MODELS = 1

train = pd.read_csv('../../..//data/train.csv')
test = pd.read_csv('../../../data/test_min.csv')

#In[]:
# sample train
train = train.sample(frac=0.01)
# In[]:

x_train = preprocess(train['comment_text'])
x_train[:2]

# In[]:
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = preprocess(test['comment_text'])


# In[ ]:

max_features = None


# In[ ]:


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)


# In[]:
x_train[:10]
# In[ ]:

max_features = max_features or len(tokenizer.word_index) + 1
max_features

# In[ ]:

crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
print('n unknown words (crawl): ', len(unknown_words_crawl))

# In[ ]:

glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
print('n unknown words (glove): ', len(unknown_words_glove))

# In[ ]:

embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
embedding_matrix.shape

del crawl_matrix
del glove_matrix
gc.collect()

# In[ ]:
f = open('emb_matrix_cat.npy','wb')
np.save(f,embedding_matrix)
f.close()
#In[]:
f = open('emb_matrix_cat.npy','rb')
embedding_matrix = np.load(f)
#In[]:
x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
#In[]:

y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()

# In[ ]:

train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = data.TensorDataset(x_test_torch)
# In[]:
all_test_preds = []
from model_utils import *
#i = 0
for model_idx in range(NUM_MODELS):
    print('Model ', model_idx)
    seed_everything(1234 + model_idx)
    
    model = NeuralNet(embedding_matrix, y_aux_train.shape[-1],max_features)
    model.cuda()
    
    test_preds = train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], 
                             loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
    all_test_preds.append(test_preds)
    print()

#In[]:
from jigsaw_metric import *
val = calc_roc(train,model,tokenizer,MAX_LEN)
print(val)
#In[]:
df = bias_metrics(train,model,tokenizer,MAX_LEN)
print(df)
# In[ ]:

submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': np.mean(all_test_preds, axis=0)[:, 0]
})

submission.to_csv('submission.csv', index=False)

#In[]:
p = get_preds(model,train_dataset,batch_size=4)

print(p.shape)
#In[]

p[0]

