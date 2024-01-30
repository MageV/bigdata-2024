import multiprocessing
import os

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from joblib import parallel_backend



df = pd.read_csv('/home/master/Downloads/songs.csv')
Y = df['artist']
X = df[['year', 'length', 'commas', 'exclamations', 'hyphens', 'colons']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
step = 0
score = 0
# while score < 0.85:
#    step += 1
#   model = DecisionTreeClassifier(max_depth=step, criterion='log_loss',
#
#                                 min_samples_split=16,class_weight='balanced')

with parallel_backend('multiprocessing'):
    param_grid = {
        'criterion': ('gini', 'log_loss', 'entropy'),
        'min_samples_split': range(2, 32),
        'max_depth': range(1, 10),
        'min_samples_leaf':range(1,10)
    }
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid,verbose=3,  scoring='f1_micro')
    grid_search.fit(X_train, Y_train)
    best_tree=grid_search.best_estimator_
    y_pred = best_tree.predict(X_test)
    score=f1_score(Y_test,y_pred,average='micro')
    print(f"GridSearchCV score_f1={score}")
    model2=DecisionTreeClassifier()
    rand_search=RandomizedSearchCV(model2,param_grid,scoring='f1_micro')
    rand_search.fit(X_train, Y_train)
    best_tree=rand_search.best_estimator_
    y_pred = best_tree.predict(X_test)
    score = f1_score(Y_test, y_pred, average='micro')
    print(f"RandSearchCV score_f1={score}")
    #    step += 1
#    if step > 20:
#        break

