import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import spectral_clustering
from sklearn.cluster import MeanShift, estimate_bandwidth

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']


def plot_pca(df, data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    principal_Components = pca.fit_transform(data)
    principal_Df = pd.DataFrame(data=principal_Components, columns=['principal component 1', 'principal component 2'])
    final_Df = pd.concat([principal_Df, df[['Type']]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    types = [1, 2, 3, 4, 5, 6, 7]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for type, color in zip(types, colors):
        indices = final_Df['Type'] == type
        ax.scatter(final_Df.loc[indices, 'principal component 1']
                   , final_Df.loc[indices, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(types)
    ax.grid()
    plt.savefig("pca.png")


def preprocessing(df, data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(0.95, whiten=True)
    principal_Components = pca.fit_transform(data)
    print("pca variance: ", pca.explained_variance_ratio_)
    return principal_Components


def kmeans(x, labels_true):
    estimator = KMeans(init='k-means++', n_clusters=7, n_init=50, max_iter=1000, n_jobs=4)
    estimator.fit(x)
    print("--------K-Means----------")
    labels = estimator.labels_
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(x, labels, metric='sqeuclidean'))


def affinity_prop(x, labels_true):
    af = AffinityPropagation(max_iter=1000).fit(x)
    labels = af.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("--------Affinity_Propagation----------")
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(x, labels, metric='sqeuclidean'))


def mean_shift(x, labels_true):
    bandwidth = estimate_bandwidth(x)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("--------Affinity_Propagation----------")
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("0 Coefficient: %0.3f"
          % metrics.silhouette_score(x, labels, metric='sqeuclidean'))


DATASET_PATH = './glass_data_labeled.csv'
df = pd.read_csv(DATASET_PATH)

# Separating out the features
x = df.loc[:, features].values

# Separating out the type
y = df.loc[:, ['Type']].values
y = y.reshape(y.shape[0])

# plot_pca(df, x)
x = preprocessing(df, x)

kmeans(x, y)
affinity_prop(x, y)
mean_shift(x, y)
