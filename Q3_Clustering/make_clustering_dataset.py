import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")

# 读取数据
q1_dataset_new = pd.read_csv('E:/25MCM\q1_dataset_new.csv')

# 选择用于聚类的特征
cluster_data = q1_dataset_new[['Year', 'NOC', 'Sport', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'gold', 'total_medal']].copy()

# 添加年份筛选功能
# start_year = int(input("请输入起始年份："))
# end_year = int(input("请输入结束年份："))
start_year, end_year = 1924, 2024

cluster_data = cluster_data[(cluster_data['Year'] >= start_year) & (cluster_data['Year'] <= end_year)]

# 去除重复行
cluster_data = cluster_data.drop_duplicates()

# 检查数据
print(cluster_data.info())

# 处理分类特征和数值特征
numeric_features = ['Year', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'gold', 'total_medal']
categorical_features = ['NOC', 'Sport', 'isHost']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)])

# 数据预处理
features = preprocessor.fit_transform(cluster_data)

# 打印处理后的特征形状
print("Transformed features shape:", features.shape)

# 自定义评分函数
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    return silhouette_score(X, labels)

# **特征降维**
# 使用 PCA 降维到指定的维度（例如 10 维）
n_components = 10  # 降到 10 维（可以根据实际情况调整）
pca = PCA(n_components=n_components)
reduced_features = pca.fit_transform(features)

# 打印降维后的特征形状
print(f"Features shape after PCA reduction: {reduced_features.shape}")

# 自动化调参和聚类
cluster_labels = {}

# KMeans 调参
param_grid_kmeans = {'n_clusters': range(2, 11)}
kmeans = KMeans(random_state=42, n_init=10)
grid_search_kmeans = GridSearchCV(estimator=kmeans, param_grid=param_grid_kmeans, scoring=make_scorer(silhouette_scorer), cv=3)
grid_search_kmeans.fit(features)
best_kmeans = grid_search_kmeans.best_estimator_
kmeans_labels = best_kmeans.fit_predict(features)
cluster_labels['KMeans'] = kmeans_labels
# 查看最佳参数和评分
print("最佳参数:", grid_search_kmeans.best_params_)
print("最佳 silhouette_score:", grid_search_kmeans.best_score_)

# Agglomerative Clustering 调参
param_grid_agglo = {'n_clusters': range(2, 11), 'linkage': ['ward', 'complete', 'average']}
agglo = AgglomerativeClustering()
grid_search_agglo = GridSearchCV(estimator=agglo, param_grid=param_grid_agglo, scoring=make_scorer(silhouette_scorer), cv=3)
grid_search_agglo.fit(features)
best_agglo = grid_search_agglo.best_estimator_
agglo_labels = best_agglo.fit_predict(features)
cluster_labels['Agglomerative'] = agglo_labels
print("Best Agglomerative parameters:", grid_search_agglo.best_params_)

# DBSCAN 调参
best_eps, best_min_samples = None, None
best_silhouette = -1
for eps in np.arange(0.3, 1.1, 0.1):
    for min_samples in range(3, 10):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 1:
            score = silhouette_score(features[non_noise_mask], labels[non_noise_mask])
            if score > best_silhouette:
                best_silhouette = score
                best_eps = eps
                best_min_samples = min_samples
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(features)
cluster_labels['DBSCAN'] = dbscan_labels
print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")

# Spectral Clustering 调参
param_grid_spectral = {'n_clusters': range(2, 11), 'affinity': ['nearest_neighbors', 'rbf']}
spectral = SpectralClustering(random_state=42)
grid_search_spectral = GridSearchCV(estimator=spectral, param_grid=param_grid_spectral, scoring=make_scorer(silhouette_scorer), cv=3)
grid_search_spectral.fit(features)
best_spectral = grid_search_spectral.best_estimator_
spectral_labels = best_spectral.fit_predict(features)
cluster_labels['Spectral'] = spectral_labels
print("Best Spectral Clustering parameters:", grid_search_spectral.best_params_)

# 计算轮廓系数
for method, labels in cluster_labels.items():
    if method != 'DBSCAN' or len(set(labels)) > 1:
        silhouette_avg = silhouette_score(features, labels)
        print(f"Silhouette Score for {method}: {silhouette_avg:.2f}")

# PCA 降维和可视化
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(features)

fig = plt.figure(figsize=(20, 10))
titles = ['KMeans', 'Agglomerative', 'DBSCAN', 'Spectral']
for i, (method, labels) in enumerate(cluster_labels.items(), start=1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels, cmap='viridis')
    ax.set_title(f'{method} Clustering')
plt.show()
