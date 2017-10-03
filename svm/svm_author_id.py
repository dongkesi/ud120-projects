#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###
from sklearn.svm import SVC
# 0.616040955631
# 0.616040955631
# 0.821387940842
# 0.892491467577
clf = SVC(C=10000.0, kernel='rbf')
t0 = time()
clf.fit(features_train, labels_train)
print 'train time:', round(time()-t0, 3), 's'
t0 = time()
acc = clf.score(features_test, labels_test)
print 'test time:', round(time()-t0, 3), 's'
print 'acc:', acc
# pred = clf.predict([features_test[10], features_test[26], features_test[50]])
pred = clf.predict(features_test)
print(pred)
cnt = 0
for i in range(len(pred)):
    if pred[i] == 1:
       cnt += 1
print cnt

#########################################################


