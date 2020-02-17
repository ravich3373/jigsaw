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
from train_utils import *
from sklearn.metrics import roc_auc_score
#import pdb

toxicity_typ = ['severe_toxicity','obscene','threat','insult','identity_attack','sexual_explicit']
eval_identities = ['male','female','homosexual_gay_or_lesbian','christian','jewish','muslim'
                  ,'black','white','psychiatric_or_mental_illness']

def calc_roc(df,model,tokenizer,maxlen,bs=512):
    #preprocess
    x = preprocess(df['comment_text'])
    y = np.where(df['target'] >= 0.5, 1, 0)
    y_aux = df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    x_ = tokenizer.texts_to_sequences(x)
    x__ = sequence.pad_sequences(x_,maxlen=maxlen)
    # create data set
    xd = torch.tensor(x__,dtype=torch.long).cuda()
    yd = torch.tensor(np.hstack([y[:,np.newaxis],y_aux]),dtype=torch.float32).cuda()

    xds = data.TensorDataset(xd,yd)
    #predict
    preds = get_preds(model,xds,batch_size=bs)
    #only need classification predictions data
    preds = preds[:,0]
    # get roc
    return roc_auc_score(y,preds),(y,preds)

def bias_metrics(df,model,tokenizer,maxlen,bs=512):
    id2roc = {}
    id2bpsn = {}
    id2bnsp = {}
    data = {}
    y = np.where(df['target'] >= 0.5, 1, 0)
    df['target'] = y
    for id in tqdm(eval_identities):
        # container fpr pred data
        d = {}
        # subgroup roc 
        df_id = df[ df[id]==1 ]
        id2roc[id],data_sg = calc_roc(df_id,model,tokenizer,maxlen,bs)
        d['sg'] = data_sg
        # bg pos sg neg roc
        df_bp = df[ (df['target']==1) & (df[id]==0) ]
        df_sn = df[ (df['target']==0) & (df[id]==1) ]
        df_bpsn = pd.concat((df_bp,df_sn), 0)
        id2bpsn[id],data_bpsn = calc_roc(df_bpsn,model,tokenizer,maxlen,bs)
        d['sg_bpsn'] = data_bpsn
        # bg neg sg pos
        df_bn = df[ (df['target']==0) & (df[id]==0) ]
        df_sp = df[ (df['target']==1) & (df[id]==1) ]
        df_bnsp = pd.concat((df_bn,df_sp), 0)
        id2bnsp[id],data_bnsp = calc_roc(df_bnsp,model,tokenizer,maxlen,bs)
        d['sg_bnsp'] = data_bnsp
        # collect sg pred data
        data[id] = d
    # create a metrics df
    met_df = pd.DataFrame()
    met_df['subgroup'] = eval_identities
    met_df['sg_auc'] = id2roc.values()
    met_df['sg_bpsn'] = id2bpsn.values()
    met_df['sg_bnsp'] = id2bnsp.values()
    # calc total roc auc
    t_roc = calc_roc(df,model,tokenizer,maxlen,bs)
    return met_df,t_roc,data
