# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:44:32 2019

@author: 真夜绫也
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


Traindata=pd.read_csv('train.csv')
Testdata = pd.read_csv('test.csv')


#Testdata['Survived'] = 0
#data = data.append(Testdata)
#PassengerId = Testdata['PassengerId']
def transform(data):
    data['Family_Size']=0
    data['Family_Size']=data['Parch']+data['SibSp']#family size
    data['Alone']=0
    data.loc[data.Family_Size==0,'Alone']=1#Alone

    data['Initial']=0
    for i in data:
        data['Initial']=data.Name.str.extract('([A-Za-z]+)\.') 
    data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

    data['Age_band']=0
    data.loc[data['Age']<=16,'Age_band']=0
    data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
    data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
    data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
    data.loc[data['Age']>64,'Age_band']=4
    data.head(2)


    data['Fare_cat']=0
    data.loc[data['Fare']<=7.91,'Fare_cat']=0
    data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
    data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

    data['Sex'].replace(['male','female'],[0,1],inplace=True)
    data['Embarked'].fillna('S',inplace=True)
    data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

    data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
    data['Fare_Range']=pd.qcut(data['Fare'],4)

#data['Fare_Range'] = label_encoder.fit_transform(data['Fare_Range'])

    data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)


"""
for i in range(len(data['Embarked'])):
    if data['Embarked'][i] == 'S':
        print(i)

f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('Family_Size','Survived',data=data,ax=ax[0])
ax[0].set_title('Family_Size vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
"""
"""
data['Fare_Range']=pd.qcut(data['Fare'],4)
print(data.groupby(['Fare_Range'])['Survived'].mean().to_frame())
"""

"""
data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

transform(Traindata)
#print(Traindata)


train,test=train_test_split(Traindata,test_size=0.3,random_state=0,stratify=Traindata['Survived'])
train_X=train.drop(['Survived'],axis=1)
train_Y=train['Survived']
test_X=test.drop(['Survived'],axis=1)
test_Y=test['Survived']
X=Traindata[Traindata.columns[1:]]
Y=Traindata['Survived']

#Y=Testdata['Survived']

transform(Testdata)
Testdata['Initial'].replace(['Dona'],1,inplace=True)
#print(Testdata)


model=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,random_state=None)

model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
#print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction1,test_Y))

model=svm.SVC(C=0.5, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)

model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
#print('Accuracy for linear SVM is',metrics.accuracy_score(prediction2,test_Y))

model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
#print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))

model=DecisionTreeRegressor(criterion="mse",splitter="best",max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.,min_impurity_split=None,presort=False)

model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
#print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction4,test_Y))

model=KNeighborsClassifier(n_neighbors=9,weights='distance') 
model.fit(train_X,train_Y)
prediction5=model.predict(Testdata)
#print('The accuracy of the KNN is',metrics.accuracy_score(prediction5,test_Y))

model=GaussianNB()
model.fit(train_X,train_Y)
prediction6=model.predict(test_X)
#print('The accuracy of the NaiveBayes is',metrics.accuracy_score(prediction6,test_Y))

model=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-09, min_samples_leaf=1,
            min_samples_split=4, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=True, random_state=13,
            verbose=0, warm_start=False)
model.fit(train_X,train_Y)
prediction7=model.predict(test_X)

from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(train_X,train_Y)
#print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))
cross=cross_val_score(ensemble_lin_rbf,X,Y, cv = 10,scoring = "accuracy")
#print('The cross validated score is',cross.mean())

predictionover = ensemble_lin_rbf.predict(Testdata)

#print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction7,test_Y))


#PassengerId = np.arange (892,1310).reshape(418,1)
#print(prediction5.shape,PassengerId.shape)

PassengerId = np.arange (892,1310).reshape(418,1)

Stackingmerge = np.c_[PassengerId,predictionover]
StackingCsv = pd.DataFrame(Stackingmerge)
StackingCsv.columns = ['PassengerId','Survived']
StackingCsv.to_csv('new2.csv',index=False,sep=',')
