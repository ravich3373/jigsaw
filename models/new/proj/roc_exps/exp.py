#In[0]:
%reload_ext autoreload
%autoreload 2
#In[1]:
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
#In[2]
y_preds = np.array()