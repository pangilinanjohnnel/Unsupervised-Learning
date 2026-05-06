import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\Johnnel\Desktop\TIP folder\1st year 2nd sem\applied machine learning\ex2\Exam score prediction\Exam_Score_Prediction.csv")

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#MODELS
print("PERFORMANCE EVALUATION")

#K-MEANS
km = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
km_score = silhouette_score(X_scaled, km.labels_)
print(f"K-Means Silhouette Score: {km_score:.4f}")

#AGGLOMERATIVE
agg = AgglomerativeClustering(n_clusters=3).fit(X_scaled)
agg_score = silhouette_score(X_scaled, agg.labels_)
print(f"Agglomerative Silhouette Score: {agg_score:.4f}")

#GMM
gmm = GaussianMixture(n_components=3, random_state=42).fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)
gmm_score = silhouette_score(X_scaled, gmm_labels)
print(f"GMM Silhouette Score: {gmm_score:.4f}")

#DBSCAN
db = DBSCAN(eps=1.5, min_samples=5).fit(X_scaled)
if len(set(db.labels_)) > 1:
    db_score = silhouette_score(X_scaled, db.labels_)
    print(f"DBSCAN Silhouette Score: {db_score:.4f}")
else:
    print("DBSCAN: Only one cluster found.")

#TESTING
print("INDIVIDUAL PREDICTIONS")

high = np.array([[21, 2, 5, 15.0, 98.0, 1, 8.0, 1, 1, 3, 3, 94.5]])
ave = np.array([[20, 1, 3, 7.0, 82.0, 1, 6.5, 2, 2, 2, 2, 75.0]])
low = np.array([[22, 1, 2, 2.0, 55.0, 0, 5.0, 3, 1, 1, 3, 42.0]])

#K-Means Pred
km_pred = km.predict(high)
print(f"K-Means says: Cluster {km_pred[0]}")
km_pred = km.predict(ave)
print(f"K-Means says: Cluster {km_pred[0]}")
km_pred = km.predict(low)
print(f"K-Means says: Cluster {km_pred[0]}")

# GMM Pred
gmm_pred = gmm.predict(high)
print(f"GMM says: Cluster {gmm_pred[0]}")
gmm_pred = gmm.predict(ave)
print(f"GMM says: Cluster {gmm_pred[0]}")
gmm_pred = gmm.predict(low)
print(f"GMM says: Cluster {gmm_pred[0]}")

#AGGLO
X_test_high = np.vstack([X_scaled, high])
X_test_ave = np.vstack([X_scaled, ave])
X_test_low = np.vstack([X_scaled, low])

agg_test = AgglomerativeClustering(n_clusters=3).fit_predict(X_test_high)
print(f"Agglomerative says: Cluster {agg_test[-1]}")
agg_test = AgglomerativeClustering(n_clusters=3).fit_predict(X_test_ave)
print(f"Agglomerative says: Cluster {agg_test[-1]}")
agg_test = AgglomerativeClustering(n_clusters=3).fit_predict(X_test_low)
print(f"Agglomerative says: Cluster {agg_test[-1]}")

#DB
X_test_high = np.vstack([X_scaled, high])
X_test_ave = np.vstack([X_scaled, ave])
X_test_low = np.vstack([X_scaled, low])

db_test = DBSCAN(eps=1.5, min_samples=5).fit_predict(X_test_high)
db_res = "Outlier (-1)" if db_test[-1] == -1 else f"Cluster {db_test[-1]}"
print(f"DBSCAN says: {db_res}")

db_test = DBSCAN(eps=1.5, min_samples=5).fit_predict(X_test_ave)
db_res = "Outlier (-1)" if db_test[-1] == -1 else f"Cluster {db_test[-1]}"
print(f"DBSCAN says: {db_res}")

db_test = DBSCAN(eps=1.5, min_samples=5).fit_predict(X_test_low)
db_res = "Outlier (-1)" if db_test[-1] == -1 else f"Cluster {db_test[-1]}"
print(f"DBSCAN says: {db_res}")

#KEYS
df['Cluster'] = km.labels_
profile = df.groupby('Cluster')[['study_hours', 'class_attendance', 'exam_score']].mean()

print("THE CLUSTER 'IDENTITY' KEY for K-MEANS")
print(profile)

df['Cluster'] = agg.labels_
profile = df.groupby('Cluster')[['study_hours', 'class_attendance', 'exam_score']].mean()

print("THE CLUSTER 'IDENTITY' KEY for AGGLO")
print(profile)

df['Cluster'] = gmm_labels
profile = df.groupby('Cluster')[['study_hours', 'class_attendance', 'exam_score']].mean()

print("THE CLUSTER 'IDENTITY' KEY for GMM")
print(profile)

df['Cluster'] = db.labels_
profile = df.groupby('Cluster')[['study_hours', 'class_attendance', 'exam_score']].mean()

print("THE CLUSTER 'IDENTITY' KEY for DBScan")
print(profile)