import warnings

import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv('/home/master/Documents/bd/diabetes.csv')
X= df[
    ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']
target_names=['w/o diabetes','diabetes']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
with parallel_backend('multiprocessing'):
    param_grid = {
        'criterion': ('gini', 'log_loss', 'entropy'),
        'min_samples_split': range(2, 32),
        'max_depth': range(1, 10),
        'min_samples_leaf': range(1, 10),
    }
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='f1_micro')
    grid_search.fit(X_train, Y_train)
    best_tree = grid_search.best_estimator_
    y_pred = best_tree.predict(X_test)
    score_1 = f1_score(Y_test, y_pred, average='micro')
    print("GridSearchCV - DecisionTree")# score={score_1}")
    print(grid_search.best_params_)
    print(f'Best score:{grid_search.best_score_}')
    print(classification_report(Y_test,y_pred,target_names=target_names))
    y_pred_proba = best_tree.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr,tpr,label=str(auc))
    plt.legend(loc=5)
    plt.show()
with parallel_backend('multiprocessing'):
    param_grid_lr = {
        'penalty': ('l1', 'l2', 'elasticnet'),
        'solver': ('saga', 'sag', 'liblinear', 'newton-cg', 'newton-cholesky'),
        'multi_class': ('ovr', 'multinomial'),
        'random_state': range(0, 5),
        'C': (0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.9, 1.0),
    }
    try:
        #train_scaler = StandardScaler().fit(X_train)
        #test_scaler = StandardScaler().fit(X_test)
        #X_train_Scaled = train_scaler.transform(X_train)
        #X_test_Scaled = test_scaler.transform(X_test)
        model = LogisticRegression()
        search_model = GridSearchCV(model, param_grid_lr, scoring='f1_micro', refit=True)
        search_model.fit(X_train, Y_train)
        best_model = search_model.best_estimator_
        y_pred = best_model.predict(X_test)
        score_1 = f1_score(Y_test, y_pred, average='micro')
        print(f"GridSearchCV - LogisticRegression")#score_f1={score_1}")
        print(f'best params:{search_model.best_params_}')
        print(f'best score:{search_model.best_score_}')
        print(classification_report(Y_test, y_pred, target_names=target_names))
    except:
        print("error")
    y_pred_proba=best_model.predict_proba(X_test)[::,1]
    fpr,tpr,_=roc_curve(Y_test,y_pred_proba)
    auc=roc_auc_score(Y_test,y_pred_proba)
    plt.plot(fpr, tpr, label=str(auc))
    plt.legend(loc=5)
    plt.show()
