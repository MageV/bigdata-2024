import numpy as np
import pandas as pd
from joblib import parallel_backend
from matplotlib import pyplot as plt
from scipy.stats import entropy
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def entropy2(p):
   return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

df = pd.read_csv('/home/master/Documents/bd/dogs.csv',index_col=0)
df.columns=['Furry','Barks','Climbs','Species']
df_X=df.drop(['Species'],axis=1)
df_Y=df['Species']
fn=np.array(['Furry','Barks','Climbs'])
cn=np.array(['cat','dog'])
X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.25, random_state=42)
with parallel_backend('multiprocessing'):
    model=DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train,Y_train)
    result=model.predict(X_test)
    score_1 = accuracy_score(Y_test, result)
    dot_data = tree.plot_tree(model,class_names=cn,feature_names=fn)
    #plt.show()
df = pd.read_csv('/home/master/Documents/bd/cats.csv',index_col=0)
df.columns=['Furry','Barks','Climbs','Species']
print(df.head())
df_X=df.drop(['Species'],axis=1)
df_Y=df['Species']
fn=np.array(['Furry','Barks','Climbs'])
X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.25, random_state=42)
with parallel_backend('multiprocessing'):
    model=DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train,Y_train)
    result=model.predict(X_test)
    score_1 = accuracy_score(Y_test, result)
    dot_data = tree.plot_tree(model,class_names=cn,feature_names=fn)
    plt.show()

