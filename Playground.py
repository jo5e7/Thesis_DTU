import numpy as np
from sklearn import metrics

metrics.roc_auc_score([0,1,0,1], [0.5, 0.6, 0.6, 0.2], 'micro')