import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_similarity_old(original_df, exfiltrated_df, numerical_cols, categorical_cols):
    print("similarity calculation")
    # Normalize ordinal and numerical attributes
    # Ensure df1 has the same number of rows as df2
    original_df = original_df.iloc[:len(exfiltrated_df)]
    original_df = original_df[exfiltrated_df.columns]
    cols = exfiltrated_df.columns
    idx = exfiltrated_df.index

    scaler = MinMaxScaler()
    original_df = scaler.fit_transform(original_df)
    exfiltrated_df = scaler.transform(exfiltrated_df)

    # Create a new DataFrame using the scaled data, original columns, and index
    original_df = pd.DataFrame(original_df, columns=cols, index=idx)
    exfiltrated_df = pd.DataFrame(exfiltrated_df, columns=cols, index=idx)

    # Calculate custom similarity
    similarities, num_similarities, cat_similarities = [], [], []
    for index, row1 in original_df.iterrows():
        row2 = exfiltrated_df.iloc[index]
        similarity = 0
        num_sim = 0
        cat_sim = 0
        n_attributes = 0

        # Compare categorical attributes
        for col in categorical_cols:
            if row1[col] == row2[col]:
                similarity += 1
                cat_sim += 1
            n_attributes += 1

        for col in numerical_cols:
            # Calculate distance for numerical attributes
            numerical_distance = abs(row1[col] - row2[col])
            similarity += 1 - numerical_distance
            num_sim += 1 - numerical_distance
            n_attributes += 1

        # Normalize similarity and express as percentage
        if similarity<0:
            similarity = 0
        if num_sim<0:
            num_sim = 0
        if cat_sim<0:
            cat_sim = 0
        similarity_percentage = (similarity / n_attributes) * 100
        num_sim_percentage = (num_sim / len(numerical_cols)) * 100
        cat_sim_percentage = (cat_sim / len(categorical_cols)) * 100
        num_similarities.append(num_sim_percentage)
        cat_similarities.append(cat_sim_percentage)
        similarities.append(similarity_percentage)
        #print(similarities)

    # Add similarity values to a new dataframe
    #similarity_df = pd.DataFrame({'Similarity (%)': similarities})
    final_cat_sim = sum(cat_similarities)/len(cat_similarities)
    final_num_sim = sum(num_similarities)/len(num_similarities)
    print("Final Categorical Similarity: ", final_cat_sim)
    print("Final Numerical Similarity: ", final_num_sim)
    final_similarity = sum(similarities)/len(similarities)
    #print(final_similarity)
    return final_similarity, final_num_sim, final_cat_sim


def calculate_similarity(original_df, exfiltrated_df, numerical_cols, categorical_cols):
    """
    Calculate bounded similarity between original and exfiltrated dataframes.
    All similarity values are guaranteed to be between 0 and 1.

    Args:
        original_df (pd.DataFrame): Original dataframe
        exfiltrated_df (pd.DataFrame): Exfiltrated dataframe
        numerical_cols (list): List of numerical column names
        categorical_cols (list): List of categorical column names

    Returns:
        tuple: (total_similarity, numerical_similarity, categorical_similarity)
    """
    # Trim original_df to match exfiltrated_df size and columns
    original_df = original_df.iloc[:len(exfiltrated_df)]
    original_df = original_df[exfiltrated_df.columns]
    cols = exfiltrated_df.columns
    idx = exfiltrated_df.index

    # Create copies to avoid modifying original data
    orig_df = original_df.copy()
    exfil_df = exfiltrated_df.copy()

    # Scale numerical columns to [0,1] range
    scaler = MinMaxScaler()
    orig_df = scaler.fit_transform(orig_df)
    exfil_df = scaler.transform(exfil_df)

    # Create a new DataFrame using the scaled data, original columns, and index
    orig_df = pd.DataFrame(orig_df, columns=cols, index=idx)
    exfil_df = pd.DataFrame(exfil_df, columns=cols, index=idx)

    similarities = []
    num_similarities = []
    cat_similarities = []

    total_cols = len(numerical_cols) + len(categorical_cols)
    num_weight = len(numerical_cols) / total_cols if total_cols > 0 else 0
    cat_weight = len(categorical_cols) / total_cols if total_cols > 0 else 0

    for index in range(len(orig_df)):
        row1 = orig_df.iloc[index]
        row2 = exfil_df.iloc[index]

        # Calculate categorical similarity (exact matches)
        if categorical_cols:
            cat_matches = sum(row1[col] == row2[col] for col in categorical_cols)
            cat_sim = cat_matches / len(categorical_cols)
            cat_similarities.append(cat_sim)
        else:
            cat_sim = 0

        # Calculate numerical similarity (bounded between 0 and 1)
        if numerical_cols:
            num_sims = []
            for col in numerical_cols:
                # Since values are scaled to [0,1], their difference is also in [0,1]
                diff = abs(row1[col] - row2[col])
                # Convert difference to similarity (1 - diff) and bound between 0 and 1
                sim = max(0.0, min(1.0, 1.0 - diff))
                num_sims.append(sim)

            num_sim = sum(num_sims) / len(numerical_cols)
            num_similarities.append(num_sim)
        else:
            num_sim = 0

        # Calculate weighted total similarity for this row
        total_sim = (num_sim * num_weight) + (cat_sim * cat_weight)
        similarities.append(total_sim)

    # Calculate final similarities (these will be between 0 and 1)
    final_cat_sim = sum(cat_similarities) / len(cat_similarities) if cat_similarities else 0
    final_num_sim = sum(num_similarities) / len(num_similarities) if num_similarities else 0

    # Calculate total similarity as weighted average
    final_similarity = (final_num_sim * num_weight) + (final_cat_sim * cat_weight)

    # Convert to percentages for printing
    print(f"\nSimilarity Results (as percentages):")
    print(f"--------------------------------")
    print(f"Numerical columns: {len(numerical_cols)} (weight: {num_weight:.2f})")
    print(f"Categorical columns: {len(categorical_cols)} (weight: {cat_weight:.2f})")
    print(f"Numerical similarity: {final_num_sim * 100:.2f}%")
    print(f"Categorical similarity: {final_cat_sim * 100:.2f}%")
    print(f"Total similarity: {final_similarity * 100:.2f}%")

    # Verify bounds
    assert 0 <= final_num_sim <= 1, f"Numerical similarity out of bounds: {final_num_sim}"
    assert 0 <= final_cat_sim <= 1, f"Categorical similarity out of bounds: {final_cat_sim}"
    assert 0 <= final_similarity <= 1, f"Total similarity out of bounds: {final_similarity}"

    return final_similarity, final_num_sim, final_cat_sim


# Example usage:
"""
# Create sample data
import pandas as pd
import numpy as np

# Sample data
original_df = pd.DataFrame({
    'num1': [1, 2, 3, 4, 5],
    'num2': [2, 4, 6, 8, 10],
    'cat1': ['A', 'B', 'A', 'B', 'A'],
    'cat2': [1, 1, 2, 2, 1]
})

exfiltrated_df = pd.DataFrame({
    'num1': [1.1, 2.2, 3.3, 4.4, 5.5],
    'num2': [2.1, 4.1, 6.1, 8.1, 10.1],
    'cat1': ['A', 'B', 'B', 'B', 'A'],
    'cat2': [1, 1, 2, 2, 2]
})

numerical_cols = ['num1', 'num2']
categorical_cols = ['cat1', 'cat2']

# Test the function
similarity, num_sim, cat_sim = verify_similarity_bounds(
    original_df, exfiltrated_df, numerical_cols, categorical_cols
)
"""


# Example usage:
"""
similarity, num_sim, cat_sim = calculate_similarity(
    original_df,
    exfiltrated_df,
    numerical_cols=['age', 'income'],
    categorical_cols=['gender', 'occupation']
)
"""