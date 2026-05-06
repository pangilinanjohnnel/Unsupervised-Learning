import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

df=pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\data mining\to do\wholesale+customers\Wholesale customers data.csv")

X = df.drop(['Channel', 'Region'], axis=1)
y = df['Channel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Reg SVC
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_dist = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

y_pred = search.predict(X_test)

print("SVC model")
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
print("Best Parameters:", search.best_params_)
print("SVC accuracy comparison")
print(f"Average CV Accuracy:{search.best_score_:.4f}")
print(f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")

#SVC w/ dim red PCA
pipe2 = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),              #PCA
    ('svm', SVC())
])

param_dist2 = {
    'pca__n_components': [2, 3, 4, 5],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

search2 = RandomizedSearchCV(
    pipe2,
    param_distributions=param_dist2,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
)

search2.fit(X_train, y_train)

y_pred2 = search2.predict(X_test)

print("SVC model PCA")
print(metrics.confusion_matrix(y_test,y_pred2))
print(metrics.classification_report(y_test,y_pred2))
print("Best Parameters:", search2.best_params_)
print("SVC model PCA accuracy comparison")
print(f"Average CV Accuracy:{search2.best_score_:.4f}")
print(f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred2):.4f}")

#SVC w/ dim red PCA and feat sel
pipe3 = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest()), #feature selection
    ('pca', PCA()),              #PCA
    ('svm', SVC())
])

param_dist3 = {
    'selector__k': [4, 5, 'all'],
    'pca__n_components': [2, 3, 4, 5],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

search3 = RandomizedSearchCV(
    pipe3,
    param_distributions=param_dist3,
    n_iter=10,
    cv=5,
    random_state=42,
    n_jobs=-1
)

search3.fit(X_train, y_train)

y_pred3 = search3.predict(X_test)

print("SVC model PCA & feature selection")
print(metrics.confusion_matrix(y_test,y_pred3))
print(metrics.classification_report(y_test,y_pred3))
print("Best Parameters:", search3.best_params_)
print("SVC model PCA & feature selection accuracy comparison")
print(f"Average CV Accuracy:{search3.best_score_:.4f}")
print(f"Test Accuracy: {metrics.accuracy_score(y_test, y_pred3):.4f}")