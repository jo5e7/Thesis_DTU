import numpy as np
from sklearn import metrics
import torch
from pprint import pprint


def get_micro_f1_score(true_labels, predicted_labels, threshold=0.5):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.round(true_labels.flatten())
    threshold = 0.5 - threshold
    predicted_labels = predicted_labels + threshold
    predicted_labels = np.round(predicted_labels.flatten())
    # print(len(true_labels))
    # print(true_labels)
    # print(np.sum(true_labels))
    # print(len(predicted_labels))
    # print(predicted_labels)
    # print(np.sum(predicted_labels))
    F1 = metrics.f1_score(true_labels, predicted_labels, average='micro', labels=[1])
    return F1

def get_macro_f1_score(true_labels, predicted_labels, weights=None, threshold=0.5):
    threshold = 0.5 - threshold
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
            #predicted_labels_per_class[i] = (predicted_labels_per_class[i]>th).float()
            #print(predicted_labels_per_class[i])
            #print(threshold)
            predicted_labels_per_class[i] = [x+threshold for x in predicted_labels_per_class[i]]
            #print(predicted_labels_per_class[i])

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

def get_micro_roc_auc_score_for_all_opacity(true_labels, predicted_labels, opacity_labels, labels_names, threshold=.5):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels.flatten()
    predicted_labels.flatten()

    opacity_labels_true = []
    opacity_labels_predicted = []
    for l in labels_names:
        opacity_labels_true.append([])
        opacity_labels_predicted.append([])

    for i in range(len(true_labels)):
        tl = true_labels[i]
        pl = predicted_labels[i]

        if (tl == 0) & (pl <= threshold):
            for j in range(len(opacity_labels_true)):
                opacity_labels_true[j].append(0)
                opacity_labels_predicted[j].append(pl)
        elif (tl == 0) & (pl > threshold):
            for j in range(len(opacity_labels_true)):
                opacity_labels_true[j].append(0)
                opacity_labels_predicted[j].append(pl)
        elif (tl == 1) & (pl <= threshold):
            for j in range(len(opacity_labels_true)):
                if opacity_labels[i][j] == 1:
                    opacity_labels_true[j].append(1)
                    opacity_labels_predicted[j].append(pl)
                else:
                    opacity_labels_true[j].append(0)
                    opacity_labels_predicted[j].append(pl)
        elif (tl == 1) & (pl > threshold):
            for j in range(len(opacity_labels_true)):
                if opacity_labels[i][j] == 1:
                    opacity_labels_true[j].append(1)
                    opacity_labels_predicted[j].append(pl)
                else:
                    opacity_labels_true[j].append(0)
                    opacity_labels_predicted[j].append(pl)

    results_dict = {}

    empty_classes = []
    all_zeros = []

    for i in range(len(opacity_labels_true)):
        print(labels_names[i])
        print(opacity_labels_true[i])
        print(opacity_labels_predicted[i])

        if (len(opacity_labels_true[i]) == 0) | (len(opacity_labels_predicted[i]) == 0):
            empty_classes.append(labels_names[i])

        if 1 not in opacity_labels_true[i]:
            all_zeros.append(labels_names[i])

    for name in empty_classes:
        empty_index = labels_names.index(name)
        del labels_names[empty_index]
        del opacity_labels_true[empty_index]
        del opacity_labels_predicted[empty_index]

    for name in all_zeros:
        empty_index = labels_names.index(name)
        del labels_names[empty_index]
        del opacity_labels_true[empty_index]
        del opacity_labels_predicted[empty_index]

    for l in range(len(labels_names)):
        print(labels_names[l])
        print(opacity_labels_true[l])
        print(opacity_labels_predicted[l])
        print(metrics.roc_auc_score(opacity_labels_true[l], opacity_labels_predicted[l], 'micro'))
        results_dict[labels_names[l]] = metrics.roc_auc_score(opacity_labels_true[l], opacity_labels_predicted[l], 'micro')
        print(results_dict)

    return results_dict

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


def get_avg_precision_score(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels.flatten()
    predicted_labels.flatten()

    avg_precision_score = metrics.average_precision_score(true_labels, predicted_labels)

    return avg_precision_score

def get_confusion_matrix(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    true_labels.flatten()
    predicted_labels.flatten()
    tn, fp, fn, tp = metrics.confusion_matrix(np.round(true_labels), np.round(predicted_labels)).ravel()

    return tn, fp, fn, tp


