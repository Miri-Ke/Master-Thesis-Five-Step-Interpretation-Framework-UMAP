
"""
What I need:
1. Dataset where rows = person/object, column = gene
2. If available, labels
"""

import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import A_Umap as fileB
import B_UnsupervisedClustering as fileC
import C_SupervisedClassification as fileD
import D_Visualization as fileE
import importlib

importlib.reload(fileD)


"""
Loading the data frames.
What I have:

Original Datasets:
original_T_gene_sample_df: This is the original dataset (that I call transposed) where row = gene (probe) and  column = brain sample (before scaling).
original_sample_gene_df: This is the transposed version of the original dataset, where row = brain sample and column = gene (probe).

Scaled Datasets:
main_scaled_sample_df: The scaled version of original_sample_gene_df, where rows = samples, and columns = genes (probes). (Probes got scaled!)
main_T_scaled_probes_df: The transposed version of the scaled dataset, where rows = genes (probes), and columns = brain samples.

Ancillary Datasets:
sample_explanation_full: Contains explanations for all possible samples in the study.
sample_this_dataset: Contains explanations specifically for the samples included in the current dataset (donor 9861).
gene_explanation: Provides explanations for all 58,692 genes (probes) included in the dataset.
"""
# the original datasets
original_T_gene_sample_df = pd.read_csv('Datasets\\normalized_microarray_donor9861\\MicroarrayExpression.csv', index_col=0, header=None)
original_sample_gene_df = original_T_gene_sample_df.T

# the scaled datasets
scaler = StandardScaler()
main_scaled_s = scaler.fit_transform(original_sample_gene_df)
main_scaled_sample_df = pd.DataFrame(main_scaled_s, columns=original_sample_gene_df.columns)
main_T_scaled_probes_df = main_scaled_sample_df.T

# the ancillary datasets
sample_explanation_full = pd.read_csv('Datasets\\normalized_microarray_donor9861\\Ontology.csv')
sample_this_dataset = pd.read_csv('Datasets\\normalized_microarray_donor9861\\SampleAnnot.csv')
probe_explanation = pd.read_csv('Datasets\\normalized_microarray_donor9861\\Probes.csv')

"""
Descriptive statistics
"""

# Flatten the DataFrame to a single series of values
all_values = original_T_gene_sample_df.values.flatten()

# Calculate statistics
absolute_min = all_values.min()
absolute_max = all_values.max()
mean_value = all_values.mean()
median_value = pd.Series(all_values).median()
std_dev = all_values.std()

# Display the results
print(f"Absolute Min: {absolute_min}")
print(f"Absolute Max: {absolute_max}")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Standard Deviation: {std_dev}")

"""
Overall Outline: 
1. Step: Run UMAP on standardized sample_gene dataset
2. Step: Unsupervised clustering on samples
Extra Step: Run clustering on genes
3. Step: Supervised Classification + Variable Selection
4. Step: Visualization.
"""

"""
1. Step: Run UMAP on standardized sample_gene dataset
'standardized_dataset': I use the standardardized dataset: main_scaled_sample_df
"""

# Run UMAP
embedding_UMAP = fileB.run_UMAP(standardized_dataset = main_scaled_sample_df, random_seed=123,
                                name_dataset=f'Brain Samples Dataset', is_labelled=f"No")

# optional: saving embedding_UMAP
with open('embedding_UMAP.pkl', 'wb') as output:
    pickle.dump(embedding_UMAP, output, pickle.HIGHEST_PROTOCOL)


# optional: loading embedding_UMAP:
with open('embedding_UMAP.pkl', 'rb') as input_file:
    embedding_UMAP = pickle.load(input_file)

plt.scatter(
    embedding_UMAP[:, 0],
    embedding_UMAP[:, 1],
    s=10,  # Size of points
    alpha=0.5
)
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP Projection of the Brain Samples Dataset', fontsize=14)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.tight_layout()
plt.show()




"""
2. Step: Run unsupervised clustering on samples to get pseudo-labels
You can adjust alpha.
"""


hier_clust = fileC.hierarchical_clustering(embedding = embedding_UMAP, n_clusters = 8, graph=True, link = 'complete')
optimal_cluster = fileC.find_optimal_clusters(embedding_UMAP, min_clusters=1, max_clusters=40, y_start=6000, num_clus=28)
gmm_labels_8 = fileC.gmm_clustering(embedding_UMAP, n_components=8, legend_small=False, seed_value=123) # seed = 123 or 222
gmm_labels_8_categorical = pd.Categorical(gmm_labels_8)


"""
Extra Step: Grouping of Probes 
"""

gmm_labels_features = fileC.gmm_clustering(main_T_scaled_probes_df, n_components=100, alpha = 0.5, plot = False, cov_type= 'spherical')

# Optionally save it:
with open('gmm_labels_features.pkl', 'wb') as output_file:
    pickle.dump(gmm_labels_features, output_file)

# Optionally: Load the object back from the computer
with open('gmm_labels_features.pkl', 'rb') as input_file:
    gmm_labels_features = pickle.load(input_file)



# Visualize the clustering results on UMAP (transposed dataset)

# Make a dataframe that has the GMM label in last column - easier for visualization.
transposed_GMM_label = main_T_scaled_probes_df.copy()
transposed_GMM_label['GMM_Labels'] = gmm_labels_features

embedding_UMAP_features = fileB.run_UMAP(standardized_dataset = main_T_scaled_probes_df, random_seed=123,
                                         name_dataset=f'Brain Samples Dataset, Showing Probes as Points',
                                         dataset_with_label = transposed_GMM_label, is_labelled=f"Yes", legend=False)



"""
3 and 4.1 Step: Feature Selection for Global Structure with Random Forest
"""
# standardized group means (so feature selection is not based on different scales):
group_means_T_standardized = fileC.meanpergroup(labels=gmm_labels_features, T_dataframe=main_T_scaled_probes_df)
group_means_standardized = group_means_T_standardized.T # group_means_T: rows = groups, but I want columns = groups!

group_means_standardized_copy = group_means_standardized.copy()
group_means_standardized_copy.columns = [' ' + str(col) for col in group_means_standardized_copy.columns]

sorted_feature_importance_8, sorted_feature_importance_dict_8 = fileD.randomforest_fun(standardized_dataset = group_means_standardized_copy, title='Feature Importances for Cluster Labels After Changing Seed',
                                                                                       labels = gmm_labels_8_categorical, max_visualized=20, label_size= 8, seed = 111) #seed 111 or 222
sorted_feature_importance_8 = [(feature.strip(), importance) for feature, importance in sorted_feature_importance_8]
sorted_features_8 = [feature for feature, _ in sorted_feature_importance_8]




"""
3 and 4.2 Step: Feature Selection for Local Structure
"""
# Changing step 2 (to have more clusters):
gmm_labels_28 = fileC.gmm_clustering(embedding_UMAP, n_components=28, legend_small=True, seed_value=123) # seed 123 or 222
gmm_labels_28_categorical = pd.Categorical(gmm_labels_28)

# Re-running step 3 (with new results from step 2)
sorted_feature_importance_28, sorted_feature_importance_dict_28 = fileD.randomforest_fun(standardized_dataset = group_means_standardized_copy,
                                                                                         labels = gmm_labels_28_categorical, max_visualized=100, label_size= 6, seed=111) # seed 111 or 222
sorted_feature_importance_28 = [(feature.strip(), importance) for feature, importance in sorted_feature_importance_28]
sorted_features_28 = [feature for feature, _ in sorted_feature_importance_28]

dict_8 = dict(sorted_feature_importance_8)
dict_28 = dict(sorted_feature_importance_28)

# Compute the difference between corresponding items in dict_28 and dict_8
difference = {key: dict_28[key] - dict_8[key] for key in dict_28.keys()}
difference_ordered_by_importance_28 = [difference[feature] for feature in sorted_features_28] #so it can be visualized in the order of feature importance according to 28 clusters



# The two Plots below show the Difference in Feature Importances After Applying a Larger Number of Clusters in the Unsupervised Clustering Step
# Plot 1 shows order sorted by importance of features after applying the larger number of clusters
plt.figure(figsize=(8, 14))
sns.barplot(x=difference_ordered_by_importance_28, y=sorted_features_28)
plt.title('Difference in Feature Importances After Applying a Larger Number \n of Clusters in the Unsupervised Clustering Step')
plt.xlabel('Importance Difference')
plt.ylabel('Feature')
plt.tick_params(axis='y', labelsize=6)  # Adjust the label size as needed
plt.show()


# Plot 2 shows the order sorted by importance of features after applying the smaller number of clusters
difference_ordered_by_importance_8 = [difference[feature] for feature in sorted_features_8]

plt.figure(figsize=(8, 14))
sns.barplot(x=difference_ordered_by_importance_8, y=sorted_features_8)
plt.title('Difference in Feature Importances After Applying a Larger Number \n of Clusters in the Unsupervised Clustering Step')
plt.xlabel('Importance Difference')
plt.ylabel('Feature')
plt.tick_params(axis='y', labelsize=6)  # Adjust the label size as needed
plt.show()


# get a list of indices that show the order from max to min in difference_ordered_by_importance_28
ordered_indices = sorted(range(len(difference_ordered_by_importance_28)),
                         key=lambda i: difference_ordered_by_importance_28[i],
                         reverse=True)

# get the probe group with the highest positive change, second highest etc.
for i in range(0, 31):
    print(sorted_features_28[ordered_indices[i]])

print(sorted_features_28[ordered_indices[0]]) # 0
print(sorted_features_28[ordered_indices[1]]) # 73
print(sorted_features_28[ordered_indices[2]]) # 91
print(sorted_features_28[ordered_indices[3]]) # 70


# When wanting to see the differences in importance ordered by absolute value:
sorted_differences = dict(sorted(difference.items(), key=lambda item: item[1], reverse=True))
diff_keys = list(sorted_differences.keys())[:20]
diff_values = list(sorted_differences.values())[:20]

plt.figure(figsize=(8, 14))
sns.barplot(x=diff_values, y=diff_keys)
plt.title('Difference in Feature Importances - Changed Seed')
plt.xlabel('Importance Difference')
plt.ylabel('Feature')
plt.tick_params(axis='y', labelsize=8)  # Adjust the label size as needed
plt.show()


"""
4. Step: Visualization
"""
# unstandardized group means for visualization
group_means_T = fileC.meanpergroup(labels=gmm_labels_features, T_dataframe=original_T_gene_sample_df)
group_means = group_means_T.T # group_means_T: rows = groups, but I want columns = groups!
group_means.columns = group_means.columns.astype(str) # for visualization


"""
Global Structure
"""

for i in range(0, 2):
    # feature_name = sorted_feature_importance[i][0]
    fileE.visualize_points(embedding=embedding_UMAP,
                           feature=sorted_feature_importance_8[i][0],
                           unlabelled_data_unstandardized=group_means,
                           data_set_name='Brain Samples Dataset',
                           legendname = 'Gene Expression Values of Probe Group Number ',
                           title='Probe Group ',
                           title_font=12)
    fig = plt.gcf()  # Get the current figure after it's been created by the function
    fig.set_size_inches(10, 5)  # Set the dimensions of the figure
    plt.show()  # Display the plot



"""
Local Structure
"""

# According to highest positive difference:
# Highest positive difference: sorted_features_28[ordered_indices[0]]
fileE.visualize_points(embedding=embedding_UMAP,
                       feature=sorted_features_28[ordered_indices[0]],
                       unlabelled_data_unstandardized=group_means,
                       data_set_name='Brain Samples Dataset',
                       legendname='Gene Expression Values of Probe Group Number ',
                       title='Probe Group ',
                       title_font=12)
fig = plt.gcf()  # Get the current figure after it's been created by the function
fig.set_size_inches(10, 5)  # Set the dimensions of the figure
plt.show()  # Display the plot

fileE.visualize_points(embedding=embedding_UMAP,
                       feature=sorted_features_28[ordered_indices[1]],
                       unlabelled_data_unstandardized=group_means,
                       data_set_name='Brain Samples Dataset',
                       legendname='Gene Expression Values of Probe Group Number ',
                       title='Probe Group ',
                       title_font=12)
fig = plt.gcf()  # Get the current figure after it's been created by the function
fig.set_size_inches(10, 5)  # Set the dimensions of the figure
plt.show()  # Display the plot




'''
Extract Probe Groups 
'''

# What do gene groups mean
## I need a background set:
all_unique_gene_symbols = probe_explanation['gene_symbol'].unique()
all_unique_gene_symbols_df = pd.DataFrame(all_unique_gene_symbols, columns=['gene_symbol'])
all_unique_gene_symbols_df.to_csv('all_unique_gene_symbols.csv', index=False, header=False)

# global structure
# Probe group 85
group_85_index = transposed_GMM_label[transposed_GMM_label['GMM_Labels'] == 85].index # get probe name
group_85_probenames = group_85_index.tolist() # make this into a list
unique_gene_symbols = probe_explanation[probe_explanation['probe_id'].isin(group_85_probenames)]['gene_symbol'].unique() # find unique gene symbols based on probe names
unique_gene_symbols_df = pd.DataFrame(unique_gene_symbols, columns=['gene_symbol'])
unique_gene_symbols_df.to_csv('group_85_unique_gene_symbols.csv', index=False, header=False) # save to csv file

# Probe group 64
group_64_index = transposed_GMM_label[transposed_GMM_label['GMM_Labels'] == 64].index # get probe name
group_64_probenames = group_64_index.tolist() # make this into a list
unique_gene_symbols = probe_explanation[probe_explanation['probe_id'].isin(group_64_probenames)]['gene_symbol'].unique() # find unique gene symbols based on probe names
unique_gene_symbols_df = pd.DataFrame(unique_gene_symbols, columns=['gene_symbol'])
unique_gene_symbols_df.to_csv('group_64_unique_gene_symbols.csv', index=False, header=False) # save to csv file

# local structure
# Probe group 0
group_0_index = transposed_GMM_label[transposed_GMM_label['GMM_Labels'] == 0].index # get probe name
group_0_probenames = group_0_index.tolist() # make this into a list
unique_gene_symbols = probe_explanation[probe_explanation['probe_id'].isin(group_0_probenames)]['gene_symbol'].unique() # find unique gene symbols based on probe names
unique_gene_symbols_df = pd.DataFrame(unique_gene_symbols, columns=['gene_symbol'])
unique_gene_symbols_df.to_csv('group_0_unique_gene_symbols.csv', index=False, header=False) # save to csv file

# Probe group 73
group_73_index = transposed_GMM_label[transposed_GMM_label['GMM_Labels'] == 73].index # get probe name
group_73_probenames = group_73_index.tolist() # make this into a list
unique_gene_symbols = probe_explanation[probe_explanation['probe_id'].isin(group_73_probenames)]['gene_symbol'].unique() # find unique gene symbols based on probe names
unique_gene_symbols_df = pd.DataFrame(unique_gene_symbols, columns=['gene_symbol'])
unique_gene_symbols_df.to_csv('group_73_unique_gene_symbols.csv', index=False, header=False) # save to csv file



