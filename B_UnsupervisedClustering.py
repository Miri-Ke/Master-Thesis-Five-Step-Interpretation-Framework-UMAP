import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

"""
Hierarchical clustering:
Meant for the embedding of UMAP (either of objects or of features)
Graph = True: Do you want the embedding graph colored by number of specified clusters? If not, then you only get dendrogram.
If yes, then you get both, dendrogram and the embedding graph (colored).

Possible 'link' methods: 
        'single': Single Linkage (minimum or nearest)
        'complete': Complete Linkage (maximum or farthest)
        'average': Average Linkage (UPGMA)
        'ward': Ward's Linkage
        'centroid': Centroid Linkage
        'median': Median Linkage
"""


def hierarchical_clustering(embedding, n_clusters, graph=True, link = 'complete', alpha = 1):
    # Perform hierarchical/agglomerative clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(embedding)

    # Plot the clustered data
    if graph:
        plt.figure()
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))  # Generate a color map with n_clusters distinct colors from tab20
        for i, color in zip(range(n_clusters), colors):
            plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], color=color, label=f'Cluster {i+1}', alpha=alpha, s=10)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title("Hierarchical Clustering")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.legend(title='Cluster Labels')  # Add a legend with a title
        plt.tight_layout()
        plt.show()


    # Generate the linkage matrix
    linked = linkage(embedding, method=link)

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True,
               color_threshold=0)
    plt.title("Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

    return labels

"""
Gaussian Mixture Models:
cov_type: This parameter determines the type of covariance parameters to use. 
It affects the volume, shape, and orientation of each Gaussian component. The possible values are:
    - 'full' (default): Each component has its own general covariance matrix. 
            This allows each Gaussian to have its own volume, shape, and orientation.
    - 'tied': All components share the same general covariance matrix. 
            This constrains all Gaussians to have the same volume, shape, and orientation.
    - 'diag': Each component has its own diagonal covariance matrix. 
            This allows each Gaussian to have its own volume but constrains the shape to be axis-aligned (no orientation).
    - 'spherical': Each component has its own single variance. This constrains each Gaussian to be spherical 
            (same variance in all directions), having the same shape and no orientation.
"""
def gmm_clustering(embedding, n_components, cov_type='full', alpha=1, plot = True, legend_small = False, m_iter = 500, seed_value = 111):
    # Instantiate and fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                          max_iter=m_iter, n_init=3, random_state=seed_value)
    gmm.fit(embedding)

    # Predict the clusters
    labels = gmm.predict(embedding)

    # Plot the clustered data
    if plot:
        plt.figure()
        # Use the same colormap as in hierarchical_clustering
        colors = plt.cm.rainbow(np.linspace(0, 1, n_components))
        for i, color in zip(range(n_components), colors):
            plt.scatter(embedding[labels == i, 0], embedding[labels == i, 1], color=color, alpha=alpha, s=10,
                        label=f'Cluster {i + 1}')

        plt.title("Gaussian Mixture Model Clusters")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        if legend_small:
            plt.legend(title="Cluster Labels", fontsize='x-small')
        else:
            plt.legend(title="Cluster Labels")
        plt.gca().set_aspect('equal', 'datalim')
        plt.show()

    return labels


'''
Find optimal number of clusters:
'''

def find_optimal_clusters(data, min_clusters, max_clusters, cov_type='full', y_start = 0, seed = 111, num_clus = None):
    bic = []
    n_components_range = range(min_clusters, max_clusters + 1)

    for n_components in n_components_range:
        # Instantiate and fit the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, random_state=seed, covariance_type=cov_type, max_iter=500)
        gmm.fit(data)

        # Append the BIC values
        bic.append(gmm.bic(data))

    # Plot the AIC and BIC values to see which n_components minimizes them
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic, label='BIC')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Bayesian Information Criterion')
    plt.ylim(bottom=y_start)
    plt.title('BIC for different numbers of clusters')
    if num_clus is not None:
        plt.axhline(y=bic[num_clus - 1], color='lightcoral', linestyle='--')
        plt.plot(num_clus, bic[num_clus - 1], 'o', color='r', markersize=8, markerfacecolor='none')
    plt.show()

    # Return the number of components that minimizes BIC
    optimal_n_components_bic = n_components_range[np.argmin(bic)]


    return optimal_n_components_bic




"""
labels = the GMM labels (NOT categorical)
data frame = The data frame that holds the numbers that should form the average.
"""

def meanpergroup(labels, T_dataframe):
    # Group the dataframe by labels
    T_dataframe['group'] = labels
    group_means = T_dataframe.groupby('group').mean()  # Calculates mean for each group

    # Drop the group column to clean up the dataframe
    T_dataframe.drop('group', axis=1, inplace=True)

    return group_means








