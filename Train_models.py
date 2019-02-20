#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# In this Script we are visualize dataset and training the models
"""
Created on Sat Feb  9 19:08:59 2019

@author: Tapsas Apostolos
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from ann_visualizer.visualize import ann_viz
from sklearn.neural_network import MLPClassifier
import numpy as np
#import Dataset
data = pd.read_excel('Dataset.xlsx')

# Put classes in y matrix
y=data['melanoma'].values
#Clear rows is not nesesary
# In means.xlsm file we do chisquare test and we find that there is no
# there in no statistically significant differences between sex and skin_canser  
data=data.drop(['eid','melanoma'],axis=1)
# Put numbers in x matrix
x=data.values

# Train svm model
clf = svm.SVC(kernel='rbf')
clf.fit(x, y) 

# Linear Regretion Model 
reg = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial').fit(x, y)

# Train Gaussian Bayes Model 
g_bayes = GaussianNB()

# Train Multinominal Bayes Model 
m_bayes = MultinomialNB()

#Train Gaussian process classification (GPC) based on Laplace approximation.
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,optimizer=None).fit(x, y) 

#Train Neural net 1 35,68
NN = MLPClassifier(activation='tanh' ,solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(1000), random_state=1) 



#Predict Models
svm_y_pred = clf.fit(x, y).predict(x)   #SVM
reg_y_pred = reg.predict(x)             #Logistic Regretion
g_bayes_y_pred = g_bayes.fit(x, y).predict(x) #Gaussian Bayes
m_bayes_y_pred = m_bayes.fit(x, y).predict(x) #Multinomina Bayes
gpc_y_pred= gpc.fit(x,y).predict(x)  #Gaussian process classification
nn_y_pred= NN.fit(x,y).predict(x)  #nn classification
# Compute confusion matrix
cnf_matrix_svm = confusion_matrix(y, svm_y_pred)
labels = ['Melanoma', 'Not_Melanoma']
plt.imshow(cnf_matrix_svm,cmap='summer')
plt.colorbar()
plt.show()
cnf_matrix_nn = confusion_matrix(y, nn_y_pred)
labels = ['Melanoma', 'Not_Melanoma']
plt.imshow(cnf_matrix_nn,cmap='summer')
plt.colorbar()
plt.show()
# SVM accuracy
svm_acc=accuracy_score(y, svm_y_pred)*100


# Logistic Regretion accuracy
reg_acc=accuracy_score(y, reg_y_pred)*100

# Gaussian Bayes accuracy
g_bayes_acc=accuracy_score(y, g_bayes_y_pred)*100

# Multinominal Bayes accuracy
m_bayes_acc=accuracy_score(y, m_bayes_y_pred)*100

#Gaussian process classification (GPC) based on Laplace approximation accuracy
gpc_acc=accuracy_score(y, gpc_y_pred)*100

#NN accuracy
nn_acc=accuracy_score(y, nn_y_pred)*100
# Plot Models Accuracies

acc=[svm_acc,nn_acc,reg_acc,g_bayes_acc,m_bayes_acc,gpc_acc]
acc_names=['svm_acc','nn_acc','reg_acc','g_bayes_acc','m_bayes_acc','gpc_acc']
plt.title("Models Accuracies")
plt.plot(acc_names,acc,'r')
plt.plot(acc_names,acc,'ko')
plt.show()
#ann_viz(NN, title="My first neural network")
# Plot preds 
k=np.arange(len(y))
k=k+1
svm_loss=cnf_matrix_svm[0][1]+cnf_matrix_svm[1][0]
svm_loss=str(svm_loss)
plt.title("SVM_ACC_POINTS")
plt.plot(k,y,'ko',label='True_Values')
plt.plot(k,svm_y_pred,'r*',label='Pred_values\n False_Values ='+svm_loss)
plt.legend(loc='best')
plt.plot()
plt.figure()
nn_loss=cnf_matrix_nn[0][1]+cnf_matrix_nn[1][0]
nn_loss=str(nn_loss)
plt.title("NN_ACC_POINTS")
plt.plot(k,y,'ko',label='True_Values')
plt.plot(k,nn_y_pred,'r*',label='Pred_values\n False_Values ='+nn_loss)
plt.legend(loc='best')
plt.plot()
bbb=NN.classes_