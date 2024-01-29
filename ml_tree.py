import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('/home/master/Downloads/songs.csv')
Y = df['artist']
X = df[['year', 'length', 'commas', 'exclamations']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
step = 0
score = 0
while score < 0.85:
    step += 1
    model = DecisionTreeClassifier(max_depth=step, criterion='gini', random_state=42,splitter='random',
                                   min_samples_split=4)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    score = f1_score(Y_test, y_pred, average='micro')
    step += 1
    if step > 20:
        break
    print(f"step:{step} score_f1={score}")
