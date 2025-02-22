import matplotlib.pyplot as plt
import umap.umap_ as umap
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches


def run_UMAP(standardized_dataset, random_seed = 1, name_dataset='Dataset Name', dataset_with_label=None,
             is_labelled = 'No', legend = True):

    """
    Runs UMAP with Default Values. If dataset has label, label should be in last column.

    Parameters:
    - standardized_dataset: pd.DataFrame. Should be standardized
    - name_dataset: Optional: Is the name of the data set

    Returns:
    - embedding: The embedding of UMAP
    """
    if is_labelled == 'Yes':
        # Factorize the last column (can be used for coloring)
        labels, unique_labels = pd.factorize(dataset_with_label.iloc[:, -1])


    reducer = umap.UMAP(random_state=random_seed)
    embedding = reducer.fit_transform(standardized_dataset)

    if is_labelled == 'Yes':
        palette = sns.color_palette("hsv", len(unique_labels))

        # Assign colors to each label
        colors = [palette[i] for i in labels]

        # Scatter plot for UMAP projection
        plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=10, alpha=0.3)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP Projection of the {name_dataset}, Colored by the GMM Label', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        # Generate legend handles using the original palette
        legend_handles = [mpatches.Patch(color=palette[i], label=f'Cluster {unique_labels[i]}')
                          for i in range(len(unique_labels))]

        # Sort handles based on the label text, which is assumed to be numeric starting after 'Cluster '
        legend_handles = sorted(legend_handles, key=lambda x: int(x.get_label().split()[-1]))

        # Add sorted legend to the plot
        if legend:
            plt.legend(handles=legend_handles, title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left',
                       fontsize='xx-small')
        plt.tight_layout()
        plt.show()


    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=10,  # Size of points
            alpha=0.5
        )
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'UMAP Projection of the {name_dataset}', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.show()



    return embedding

