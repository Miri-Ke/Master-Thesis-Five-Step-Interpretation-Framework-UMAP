import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import A_Umap as fileB
import B_UnsupervisedClustering as fileC
import C_SupervisedClassification as fileD
import D_Visualization as fileE


'''
Step 0: Data Preprocessing
'''

# Load the dataset
df_full = pd.read_csv('Datasets\\Melbourne_housing_original.csv')

df_full.rename(columns={'SellerG': 'Real Estate Agent', 'Type': 'Residence Type', 'Method': 'Selling Method',
                        'CouncilArea': 'Council Area', 'Regionname': 'Region Name', 'Distance': 'Distance to CBD',
                        'Car': 'Parking Spaces', 'Landsize': 'Land Size', 'BuildingArea': 'Building Area',
                        'YearBuilt': 'Year Built', 'Lattitude': 'Latitude', 'Longtitude': 'Longitude',
                        'Propertycount': 'Residences in Suburb', 'Bedroom2': 'Number of Bedrooms',
                        'Rooms': 'Number of Rooms', 'Bathroom': 'Number of Bathrooms'}, inplace=True)

df_new = df_full.drop(['Address', 'Suburb', 'Real Estate Agent', 'Postcode', 'Date'], axis=1)

# Drop rows with missing values
df = df_new.dropna()


replacements = {'h': 'House', 't': 'Townhouse', 'u': 'Unit'}
df.loc[:, 'Residence Type'] = df['Residence Type'].replace(replacements)

# Define variable types
categorical_vars = ['Residence Type', 'Selling Method', 'Council Area', 'Region Name']
numerical_vars = ['Number of Rooms', 'Price', 'Distance to CBD', 'Number of Bathrooms', 'Parking Spaces', 'Land Size', 'Building Area', 'Year Built', 'Latitude',
                  'Longitude', 'Residences in Suburb', 'Number of Bedrooms']

# Standardize numerical variables
scaler = StandardScaler()
df_standardized_numerical = pd.DataFrame(scaler.fit_transform(df[numerical_vars]), columns=numerical_vars, index=df.index)

# Dummy encode categorical variables
one_hot = OneHotEncoder()
df_onehot_categorical = pd.DataFrame(one_hot.fit_transform(df[categorical_vars]).toarray(), columns=one_hot.get_feature_names_out(), index=df.index)

# Combine standardized numerical and dummy encoded categorical data
onehot_standardized_df = pd.concat([df_standardized_numerical, df_onehot_categorical], axis=1)

# For the DataFrame with original data
original_numerical = df[numerical_vars]  # Already a DataFrame with original unstandardized data
original_categorical = df[categorical_vars].astype('category')

# Combine original numerical and categorical data
original_df = pd.concat([original_numerical, original_categorical], axis=1)
print(original_df['Selling Method'].unique())

# DataFrame with standardized numerical and original categorical variables
stand_and_orig_cat = pd.concat([df_standardized_numerical, original_categorical], axis=1)

""" 
Now we have:
`original_df` - DataFrame with original unstandardized numerical and original categorical variables
`onehot_standardized_df` - DataFrame with standardized numerical and one-hot encoded categorical variables
'stand_and_orig_cat' - DataFrame with standardized numerical and original categorical variables

'original_numerical' - DataFrame with only the unstandardized numerical variables
'df_standardized_numerical' - DataFrame with only the standardized numerical variables
'original_categorical' - DataFrame with only the original categorical variables
'df_onehot_categorical' - DataFrame with only the one-hot encoded categorical variables
"""


'''
Step 1: UMAP
'''

embedding_UMAP = fileB.run_UMAP(standardized_dataset = onehot_standardized_df, random_seed=111,
                                name_dataset=f'Melbourne Housing Data Set', is_labelled=f"No")

'''
Step 2: Unsupervised Clustering
'''

hier_clust = fileC.hierarchical_clustering(embedding = embedding_UMAP, n_clusters = 5, graph=True, link = 'complete')
hier_labels_categorical = pd.Categorical(hier_clust)

gmm_labels = fileC.gmm_clustering(embedding_UMAP, n_components=5, seed_value = 222)
gmm_labels_categorical = pd.Categorical(gmm_labels)


'''
Step 3: Supervised Classification
'''

log_regression_results = fileD.logregression(dataset = stand_and_orig_cat, label = hier_labels_categorical, sort_by ='BIC', seed = 222)


## visualizing the log regression results
variable_names = log_regression_results.iloc[:, 0].reset_index(drop=True)
f1_scores = log_regression_results.iloc[:, 1]
bic_scores = log_regression_results.iloc[:, 3]

# Creating the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))
bic_line = ax1.plot(variable_names, bic_scores, color='red', marker='o', label='BIC Score')
ax1.set_ylim(0, max(bic_scores) * 1.1)
ax2 = ax1.twinx()
f1_line = ax2.plot(variable_names, f1_scores, color='darkblue', marker='1', label='F1 Score', linewidth=0.8)
ax2.set_ylim(max(f1_scores) * 1.1, 0)

ax1.set_xlabel('Variables')
ax1.set_ylabel('BIC Score')
ax2.set_ylabel('F1 Score')
plt.title('BIC Scores and F1 Scores for Logistic Regression Variables')
ax1.set_xticks(range(len(variable_names)))
ax1.set_xticklabels(variable_names, rotation=70, ha="right")

handles, labels = [], []
axes = [ax1, ax2]
for ax in axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:  # check if label is already in the list
            handles.append(handle)
            labels.append(label)
ax1.legend(handles, labels, loc='upper left')

plt.tight_layout()  # Adjust layout to fit everything neatly
plt.show()


'''
Step 4: Visualization
'''

fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Region Name',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       colormap='tab20')
fig = plt.gcf()
fig.set_size_inches(10, 8)  # Set the dimensions of the figure


fileE.visualize_points(embedding=embedding_UMAP,
                       feature= 'Longitude',
                       legendname='',
                       unlabelled_data_unstandardized=original_df,
                       data_set_name='Melbourne Housing Dataset',
                       colormap='RdYlBu_r')
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()  # Display the plot


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Council Area',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       legendfont='xx-small',
                       colormap= 'rainbow')
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points(embedding=embedding_UMAP,
                       feature= 'Latitude',
                       legendname='',
                       unlabelled_data_unstandardized=original_df,
                       data_set_name='Melbourne Housing Dataset',
                       colormap='RdYlBu_r')
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()


fileE.visualize_points(embedding=embedding_UMAP,
                       feature= 'Distance to CBD',
                       legendname='',
                       unlabelled_data_unstandardized=original_df,
                       data_set_name='Melbourne Housing Dataset',
                       colormap='RdYlBu_r')
fig = plt.gcf()  # Get the current figure after it's been created by the function
fig.set_size_inches(10, 8)  # Set the dimensions of the figure
plt.show()  # Display the plot


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Price',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       max_cap_value= 4000000,
                       note_x=0.85,
                       note_y = 0.97,
                       cap_info_location='text_box',
                       colormap='RdYlBu_r')
fig = plt.gcf()
fig.set_size_inches(10, 8)



fileE.visualize_points(embedding=embedding_UMAP,
                       feature= 'Residences in Suburb',
                       legendname='',
                       unlabelled_data_unstandardized=original_df,
                       data_set_name='Melbourne Housing Dataset',
                       colormap='RdYlBu_r')
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Year Built',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       min_cap_value=1850,
                       colormap='RdYlBu_r',
                       cap_info_location='text_box')
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points_discrete(embedding=embedding_UMAP,
                                feature= 'Number of Bedrooms',
                                legendname='',
                                unlabelled_data_unstandardized=original_df,
                                data_set_name='Melbourne Housing Dataset',
                                max_cap_value=7)
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points_discrete(embedding=embedding_UMAP,
                                feature= 'Number of Rooms',
                                legendname='',
                                max_cap_value=7,
                                unlabelled_data_unstandardized=original_df,
                                data_set_name='Melbourne Housing Dataset',
                                note_y = 0.7)
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Building Area',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       max_cap_value=500,
                       colormap='RdYlBu_r',
                       cap_info_location='text_box')
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points_discrete(embedding=embedding_UMAP,
                                feature= 'Number of Bathrooms',
                                legendname='',
                                max_cap_value=5,
                                unlabelled_data_unstandardized=original_df,
                                data_set_name='Melbourne Housing Dataset',
                                note_y = 0.76)
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Land Size',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       max_cap_value=4000,
                       colormap='RdYlBu_r',
                       cap_info_location='text_box')
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points_discrete(embedding=embedding_UMAP,
                                feature= 'Parking Spaces',
                                legendname='',
                                unlabelled_data_unstandardized=original_df,
                                data_set_name='Melbourne Housing Dataset',
                                max_cap_value=8,
                                note_y= 0.64)
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Residence Type',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset')
fig = plt.gcf()
fig.set_size_inches(10, 8)


fileE.visualize_points(embedding = embedding_UMAP,
                       feature = 'Selling Method',
                       legendname='',
                       unlabelled_data_unstandardized = original_df,
                       data_set_name = 'Melbourne Housing Dataset',
                       colormap='rainbow',
                       transparency=0.3)
fig = plt.gcf()
fig.set_size_inches(10, 8)






