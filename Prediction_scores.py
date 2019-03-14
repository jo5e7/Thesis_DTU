import numpy as np
from sklearn import metrics
import torch


def get_micro_f1_score(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.round(true_labels.flatten())
    predicted_labels = np.round(predicted_labels.flatten())
    # print(len(true_labels))
    # print(true_labels)
    # print(np.sum(true_labels))
    # print(len(predicted_labels))
    # print(predicted_labels)
    # print(np.sum(predicted_labels))
    F1 = metrics.f1_score(true_labels, predicted_labels, average='micro', labels=[1])
    return F1

def get_macro_f1_score(true_labels, predicted_labels, weights=None):
    true_labels_per_class = []
    predicted_labels_per_class = []
    score_per_class = []
    for i in range(len(true_labels[0])):
        true_labels_per_class.append([])
        predicted_labels_per_class.append([])

    for l in true_labels:
        for i in range(len(l)):
            true_labels_per_class[i].append(l[i])

    for l in predicted_labels:
        for i in range(len(l)):
            predicted_labels_per_class[i].append(l[i])

    for i in range(len(true_labels_per_class)):
        # print(true_labels_per_class[i])
        # print(np.round(predicted_labels_per_class[i]))
        try:
            score_per_class.append(
                metrics.f1_score(true_labels_per_class[i], np.round(predicted_labels_per_class[i]), average='micro', labels=[1]))
        except ValueError:
            print('ValueError, class', i, 'do not have positive samples and F1 score is not defined in that case.')
            pass

    if weights is None:
        return score_per_class
    else:
        return weights * score_per_class

def get_micro_roc_auc_score(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels.flatten()
    predicted_labels.flatten()
    score = metrics.roc_auc_score(true_labels, predicted_labels, 'micro')
    return score

def get_macro_roc_auc_score(true_labels, predicted_labels, weights = None):
    #true_labels = np.array(true_labels)
    #predicted_labels = np.array(predicted_labels)
    #score_per_class = np.zeros(len(true_labels))

    true_labels_per_class = []
    predicted_labels_per_class = []
    score_per_class = []
    for i in range(len(true_labels[0])):
        true_labels_per_class.append([])
        predicted_labels_per_class.append([])

    for l in true_labels:
        for i in range(len(l)):
            true_labels_per_class[i].append(l[i])

    for l in predicted_labels:
        for i in range(len(l)):
            predicted_labels_per_class[i].append(l[i])


    for i in range(len(true_labels_per_class)):
        # print(true_labels_per_class[i])
        # print(predicted_labels_per_class[i])
        try:
            # score_per_class[i] = metrics.roc_auc_score(true_labels_per_class[i], predicted_labels_per_class[i], 'micro')
            score_per_class.append(metrics.roc_auc_score(true_labels_per_class[i], predicted_labels_per_class[i], 'micro'))
        except ValueError:
            print('ValueError, class', i, 'do not have positive samples and ROC AUC score is not defined in that case.')
            pass

    if weights is None:
        return score_per_class
    else:
        return weights * score_per_class



