import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_stacked_vals_per_cat(df, feature):
    label_counts_per_category = df.groupby(feature)['label'].value_counts().unstack(fill_value=0)
    max_row = max(label_counts_per_category.sum(axis=1))
    categories = label_counts_per_category.index.tolist()
    x_pos = np.arange(len(label_counts_per_category))

    bar_width = 0.5
    colors = ['green', 'purple']
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(label_counts_per_category)):
        label_counts = label_counts_per_category.iloc[i]
        edible_count = label_counts[0]
        poisonous_count = label_counts[1]
        ax.bar(x_pos[i], edible_count, width=bar_width, label='edible', color=colors[0])
        ax.bar(x_pos[i], poisonous_count, width=bar_width, bottom=edible_count, label='poisonous', color=colors[1])
            # Add legend and labels
    ax.set_ylabel('Counts of edible vs poisonous')
    ax.set_xlabel(f'Categories in {feature}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_yticks(np.arange(0, max_row+1, 1))
    labels=['edible', 'poisonous']
    ax.legend(labels, loc='upper right', fancybox=True, framealpha=0.8, facecolor='white', edgecolor='gray', title='Legend')

    #loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(categories), frameon=True, framealpha=0.8,
              #fancybox=True, fontsize='small', borderaxespad=0.25
    # Show the plot
    plt.savefig(f'stacked_barplot_{feature}.png')
    plt.close()

# Example dataframe
df = pd.DataFrame({'cap_shape': ['A', 'A', 'B', 'B', 'B', 'C', 'C', 'D', 'B', 'C', 'C'],
                   'cap_surface': ['m', 'f', 'g', 'f', 'm', 'm', 'g', 'g', 'm', 'g', 'f'],
                   'label': ['p', 'p', 'p', 'e', 'e', 'e', 'p', 'e', 'e', 'p', 'e']})


col_names = df.columns
feature_names = list(col_names[:-1])

for feature in feature_names:
    plot_stacked_vals_per_cat(df, feature)
