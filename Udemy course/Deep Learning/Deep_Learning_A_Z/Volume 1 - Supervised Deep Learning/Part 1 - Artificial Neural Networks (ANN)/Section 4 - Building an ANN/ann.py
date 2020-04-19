import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

labelencoder_X_1=LabelEncoder()
labelencoder_X_2=LabelEncoder()

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]



ct = ColumnTransformer([("Geography",OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])


X=X[:,1:]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)



import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,kernel_initializer='uniform',
                         activation='relu',
                        input_dim=11))
    classifier.add(Dense(output_dim=6,kernel_initializer='uniform',
                         activation='relu'
                         ))
    classifier.add(Dense(output_dim=1,kernel_initializer='uniform',
                         activation='sigmoid',
                        ))
    classifier.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,
                          y=y_train,cv=10,n_jobs=5)





















