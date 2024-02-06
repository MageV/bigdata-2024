import matplotlib.pyplot as plt
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('/home/master/Documents/bd/heart_.csv')
X=df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
Y=df["condition"]
model=RandomForestClassifier(n_estimators=10,max_depth=5)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,shuffle=True)
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
score_1 = f1_score(Y_test, y_pred, average='micro')
print(f"F1 score={score_1}")
print(model.feature_importances_)
sns.heatmap(X.corr(),annot=True,fmt='.2f',cmap='Blues',cbar=None)
plt.show()