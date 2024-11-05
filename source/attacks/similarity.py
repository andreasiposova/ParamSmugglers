import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calculate_similarity(original_df, exfiltrated_df, numerical_cols, categorical_cols):
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