import numpy as np
import matplotlib.pyplot as plt


def visualize_points(embedding, feature, unlabelled_data_unstandardized, data_set_name, legendname,
                              max_cap_value=None, min_cap_value=None, title=None, title_font=12, legendfont='medium',
                              colormap='rainbow', transparency=0.7, note_x=0.87, note_y=0.97, cap_info_location='label'):
    is_categorical = unlabelled_data_unstandardized[feature].dtype.name == 'category'

    # Create the plot
    plt.figure()

    if is_categorical:
        # Handle categorical data
        important_feature = unlabelled_data_unstandardized[feature].cat.codes
        categories = unlabelled_data_unstandardized[feature].cat.categories
        n_categories = len(categories)
        cmap = plt.get_cmap(colormap, n_categories)  # Use a discrete colormap with enough colors

        # Scatter plot for each category
        for i, category in enumerate(categories):
            idx = unlabelled_data_unstandardized[feature] == category
            plt.scatter(embedding[idx, 0], embedding[idx, 1], color=cmap(i),
                        edgecolors='k', alpha=transparency, s=50, label=str(category))

        # Legend for categorical data
        plt.legend(title=f'{legendname}{feature}', loc='upper left', fontsize=legendfont)
    else:
        # Handle continuous data optionally with cap values
        original_feature = unlabelled_data_unstandardized[feature].values
        important_feature = original_feature.copy()
        cap_text = ''
        note_text = ''
        if max_cap_value is not None or min_cap_value is not None:
            # Clip the important_feature
            important_feature = np.clip(important_feature, min_cap_value, max_cap_value)
            if cap_info_location == 'label':
                # Prepare cap_text
                if min_cap_value is not None and max_cap_value is not None:
                    cap_text = f' (capped at {min_cap_value} to {max_cap_value})'
                elif max_cap_value is not None:
                    cap_text = f' (capped at {max_cap_value}+)'
                elif min_cap_value is not None:
                    cap_text = f' (capped at {min_cap_value}-)'
            elif cap_info_location == 'text_box':
                # Prepare note_text
                max_value_before_cap = int(np.max(original_feature))
                min_value_before_cap = int(np.min(original_feature))
                if max_cap_value is not None:
                    note_text += f'Capped at {max_cap_value}+, \nOriginal max: {max_value_before_cap}'
                if min_cap_value is not None:
                    if note_text != '':
                        note_text += '\n'
                    note_text += f'Lower bound at {min_cap_value}-, \nOriginal min: {min_value_before_cap}'
            else:
                # cap_info_location is neither 'label' nor 'text_box', ignore cap info
                pass

        # Plot the scatter plot
        cmap = plt.get_cmap('RdYlBu_r')
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=important_feature, cmap=cmap, edgecolors='k', alpha=transparency, s=50)
        colorbar = plt.colorbar(scatter, label=f'{legendname}{feature}{cap_text}')

        if cap_info_location == 'text_box' and note_text != '':
            # Add text box
            plt.gca().text(note_x, note_y, note_text, transform=plt.gca().transAxes, fontsize=10,
                           verticalalignment='top', horizontalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black'))

    # Setting the title and labels
    title_text = f"Scatter Plot of {title}{feature} on UMAP's Embedding for the {data_set_name}" if title else f"Scatter Plot of {feature} on UMAP's Embedding for the {data_set_name}"
    plt.title(title_text, fontsize=title_font)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.gca().set_facecolor('#f0f0f0')
    plt.gca().set_aspect('equal', 'datalim')
    plt.tight_layout()
    plt.show()



def visualize_points_discrete(embedding, feature, unlabelled_data_unstandardized, data_set_name, legendname, max_cap_value, title=None, title_font=12, legendfont ='medium',
                              transparency = 0.7, note_x = 0.915, note_y = 0.66):
    # Extracting feature values for numerical data
    important_feature = unlabelled_data_unstandardized[feature].values

    max_value_before_cap = int(np.max(important_feature))
    min_value_before_cap = int(np.min(important_feature))

    # Cap feature values
    capped_feature = np.clip(important_feature, None, max_cap_value)
    unique_features = np.unique(capped_feature)
    cmap = plt.get_cmap('RdYlBu_r')
    colors = cmap(np.linspace(0, 1, len(unique_features)))
    color_dict = dict(zip(unique_features, colors))

    # Map feature values to colors
    color_mapped = [color_dict[val] for val in capped_feature]

    # Creating the plot
    plt.figure()

    # For discrete data, use a single scatter plot
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=color_mapped,
                          edgecolors='k', alpha=transparency, s=50)

    # Create a custom legend with integer labels and edge colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='k', markersize=8) for c in colors]
    labels = [f'{legendname}{int(v)}' for v in unique_features]
    if max_cap_value in unique_features:
        labels[-1] = f'{legendname}{max_cap_value}+'  # Label for cap value and above
    plt.legend(handles, labels, title=f'{legendname}{feature}', fontsize=legendfont)

    note_text = ''
    if max_cap_value is not None:
        note_text += f'Capped at {max_cap_value}+, \n Original max: {max_value_before_cap}'

        plt.gca().text(note_x, note_y, note_text, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='black'))


    # Setting the title and labels
    title_text = f"Scatter Plot of {title}{feature} on UMAP's Embedding for the {data_set_name}" if title else f"Scatter Plot of {feature} on UMAP's Embedding for the {data_set_name}"
    plt.title(title_text, fontsize=title_font)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    plt.gca().set_facecolor('#f0f0f0')
    plt.gca().set_aspect('equal', 'datalim')
    plt.tight_layout()
    plt.show()

