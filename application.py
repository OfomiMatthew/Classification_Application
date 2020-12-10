# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:44:47 2020

@author: User
"""
#IMPORTING THE DIFFERENT LIBARIES
import numpy as np

import pandas as pd
from sklearn import datasets
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("Classification Algorithm App")
st.write("""
 # Explore the classifier
 which is the best?
 """)
dataset_name = st.sidebar.selectbox("Select dataset", ("Iris data","Breast cancer"))
st.write(dataset_name)

classifier_name = st.sidebar.selectbox("Select classifier", ("Logistic regression","SVM","KNN"))

def get_dataset(dataset_name):
    if dataset_name == "Iris data":
        data=datasets.load_iris()
    else:
        data =datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X,y
X,y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

def parameter(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1 , 15)
        params["K"] = K
    elif clf_name =="SVM":
        C = st.sidebar.slider("C",0.1, 100.0)
        gamma = st.sidebar.slider("gamma", 0.1, 100.0)
        params["C"] = C
        params["gamma"]= gamma
    else:
        C = st.sidebar.slider("C",0.1, 10.0)
        params["C"] = C
    
    
    return params
params = parameter(classifier_name)

def get_classifier(clf_name,params): #defining the classifier function
    if clf_name == "KNN":
         
       clf = KNeighborsClassifier(n_neighbors=params["K"])
      
    elif clf_name =="SVM":
       
       clf = SVC(C=params["C"],gamma=params["gamma"])
       
    else:
        
        clf = LogisticRegression(C=params["C"])
        
    return clf
clf = get_classifier(classifier_name,params)

#performing classification
X_train,X_test,y_train,y_test = train_test_split (X,y, test_size=0.20,random_state=42)   
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#Performing accuracy of the models
acc = accuracy_score(y_test,y_pred)   

st.write(f"classifier ={classifier_name}")
st.write(f"accuracy ={acc}")  
       
#plotting the dataset
pca = PCA(n_components=2) #USING PCA TO CONVERT DATA TO 2-DIMENSION
X_projected=pca.fit_transform(X) 
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.subplot()
plt.scatter(x1,x2, c=y, alpha=0.75, cmap="viridis")
plt.xlabel("component_1")
plt.ylabel("component_2")
plt.colorbar()
st.pyplot(fig)





