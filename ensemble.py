import os
import pandas as pd
import numpy as np
import statistics
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from config_shared import *
import sys
from collections import Counter

# read data
first = True
for filename in os.listdir(test_predictions_prob_dir):
    print(filename)
    if filename.endswith(".csv"):
        if first:
            Xp = np.array(pd.read_csv(test_predictions_prob_dir + filename, index_col=0))
            first = False
        else:
            X2p = np.array(pd.read_csv(test_predictions_prob_dir + filename, index_col=0))
            Xp = np.concatenate((Xp, X2p), axis=1)

first = True
for filename in os.listdir(val_predictions_prob_dir):
    print(filename)
    if filename.endswith(".csv"):
        if first:
            Xv = np.array(pd.read_csv(val_predictions_prob_dir + filename, index_col=0))
            first = False
        else:
            X2v = np.array(pd.read_csv(val_predictions_prob_dir + filename, index_col=0))
            Xv = np.concatenate((Xv, X2v), axis=1)

Yv = np.array(pd.read_csv(val_true_labels_dir + dataset_version + 'validation_labels' + '.csv', index_col=0)).ravel()

print(Xp.shape, Xv.shape, Yv.shape)


def predictCV(model, par, name, val=Xv, test=Xp):
    print('---------------- '+name+' -------------------')
    clf = GridSearchCV(model, par, return_train_score=True, cv=5, scoring="accuracy", refit=True, n_jobs=-1)
    output = clf.fit(val, Yv)
    print(output)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Grid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.5f (+/-%0.05f) for %r"
              % (mean, std * 2, params))
    pred = np.array(clf.predict(test))
    testY = pd.DataFrame(pred, columns=["Prediction"])
    testY.index.name = 'Id'
    testY.index += 1
    testY.to_csv(ens_pred + name + '.csv')

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

def models(name):
    if (name == "svc"):
        parameters = [{'kernel':['linear'], 'C':[0.01, 0.05, 0.1, 0.5, 1]}]
        predictCV(SVC(), parameters, "svc")

    if (name == "lr"): 
        parameters = [
            {'penalty': ['l1'], 'max_iter': [1e8], 'C': [0.0001, 0.001, 0.01, 0.1, 1], 'solver': ['liblinear']},
            {'penalty': ['l2'], 'max_iter': [1e8], 'C': [0.0001, 0.001, 0.01, 0.1, 1],
            'solver': ['newton-cg', 'lbfgs', 'sag', 'saga']},
            {'penalty': ['elasticnet'], 'max_iter': [1e8], 'C': [0.0001, 0.001, 0.01, 0.1, 1], 'solver': ['saga'],
            'l1_ratio': [0.5]}
        ]
        predictCV(LogisticRegression(), parameters, "lr")

    if (name == "gnb"):
        predictCV(GaussianNB(), [{}], name)

    if (name == "bag"):
        predictCV(BaggingClassifier(base_estimator=LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=1e8), n_estimators=500), [{}], name+'500')

    estim = [('lr', LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=1e8)), ('svc', SVC(kernel='linear', C=0.05, probability=True)), ('gnb', GaussianNB())] 
    estim2 = [('lr', LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=1e8)), ('svc', SVC(kernel='linear', C=0.05, probability=True)), ('bag', BaggingClassifier(base_estimator=LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=1e8), n_estimators=500))] 
    
    if (name == "vc"):
        predictCV(VotingClassifier(estimators=estim), [{'voting' : ['soft', 'hard']}], name+'gnb')
        predictCV(VotingClassifier(estimators=estim2), [{'voting' : ['soft', 'hard']}], name+'bag')

    if (name == "majority-soft"):
        X = np.average(Xv, axis=1)
        correct = 0
        for x,y in zip(X,Yv):
            if (x <= 0.5 and y == 1) or (x > 0.5 and y == -1):
                correct += 1
        
        print("Validation accuracy:", correct / Xv.shape[0])
        pred = []
        X = np.average(Xp,axis=1)
        for x in X:
            pred.append(1 if x < 0.5 else -1)
        testY = pd.DataFrame(pred, columns=["Prediction"])
        testY.index.name = 'Id'
        testY.index += 1
        testY.to_csv(ens_pred + name + '.csv')
    
    if (name == "majority-hard"):
        Xp[Xp >= 0.5] = 2
        Xp[Xp < 0.5] = 1
        Xp[Xp == 2] = -1

        Xv[Xv >= 0.5] = 2
        Xv[Xv < 0.5] = 1
        Xv[Xv == 2] = -1
        correct = 0
        for x,y in zip(Xv,Yv):
            y_pred = Counter(x).most_common()[0][0]
            #print(x, "=>", y_pred)
            if int(y_pred) == y:
                correct += 1
        print("Accuracy:", correct / Xv.shape[0])
        pred = []
        for x in Xp:
            pred.append(int(Counter(x).most_common()[0][0]))
        testY = pd.DataFrame(pred, columns=["Prediction"])
        testY.index.name = 'Id'
        testY.index += 1
        testY.to_csv(ens_pred + name + '.csv')
    return

models(sys.argv[1])
