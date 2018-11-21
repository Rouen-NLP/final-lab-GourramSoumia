#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:37:52 2018

@author: gourram
"""

import seaborn as sns
import pandas as pd
import os 

#%% Données

data = pd.read_csv("Tobacco3482.csv")
print (" Le nombre d'article ", len(data))

# Plot the statistics of label
sns.countplot(data=data, y = 'label')


#%%
 
# Récupération des fichiers
Classes = os.listdir('data/Tobacco3482-OCR' )

text=[]
Contenus=[]
label=[]

# Le nombre de classes
c = len(Classes)

for i in range(c):
    # Récupération des fichiers txt de chaque classe
    Contenus.append(os.listdir('data/Tobacco3482-OCR/%s' %Classes[i]))
    
    # OUvrir les fichiers txt et récupérer le texte  
    for j in range(len(Contenus[i])):
       op=open('data/Tobacco3482-OCR/%s/%s' %(Classes[i],Contenus[i][j]), encoding="utf8")
       text.append(op.read())
       label.append(Classes[i])


#%% Création d'une dataframe
       
DF=pd.DataFrame()
DF['text'] = text
DF['label'] = label

#%%

from sklearn.model_selection import train_test_split


X_train,X_test, y_train,y_test = train_test_split(DF.text, DF.label, test_size=0.20, 
                                                random_state=1)

X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.20, 
                                                random_state=1)

print('nb apprentissage :' ,X_train.shape)
print('nb test:', X_test.shape)
print('nb validation:', X_dev.shape)

#%% Vectorisation

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Vectorisation
vectorizer = CountVectorizer(max_features=3000)
vectorizer.fit(X_train)

X_train_vect = vectorizer.transform(X_train)
X_test_vect= vectorizer.transform(X_test)
X_dev_vect= vectorizer.transform(X_dev)

# Représentation TF_IDF
tf_transformer = TfidfTransformer().fit(X_train_vect)

# transformation tf-idf des ensemble train et test
X_train_tf = tf_transformer.transform(X_train_vect)
X_test_tf = tf_transformer.transform(X_test_vect)
X_dev_tf = tf_transformer.transform(X_dev_vect)

#%%

# Entrainement avec un classifieur MLP
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Definition du classifieur
classifier= MLPClassifier(alpha = 1)
classifier.fit(X_train_vect, y_train)

# Observasion de la précision
predictions = classifier.predict(X_dev_vect)
#acc_test= classifier.score(X_test_vect, y_test)

accuracy = accuracy_score(predictions, y_dev)
print('Le score obtenu:',accuracy*100,'%' )

 