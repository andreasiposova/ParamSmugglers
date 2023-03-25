import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def calculate_similarity(original_df, exfiltrated_df, numerical_cols, categorical_cols):
    # Normalize ordinal and numerical attributes
    # Ensure df1 has the same number of rows as df2
    original_df = original_df.iloc[:len(exfiltrated_df)]
    original_df = original_df[exfiltrated_df.columns]

    #scaler = MinMaxScaler()
    #original_df[numerical_cols] = scaler.fit_transform(original_df[numerical_cols])
    #exfiltrated_df[numerical_cols] = scaler.transform(exfiltrated_df[numerical_cols])

    # Calculate custom similarity
    similarities = []
    for index, row1 in original_df.iterrows():
        row2 = exfiltrated_df.iloc[index]
        similarity = 0
        n_attributes = 0

        # Compare categorical attributes
        for col in categorical_cols:
            if row1[col] == row2[col]:
                similarity += 1
            n_attributes += 1

        for col in numerical_cols:
            # Calculate distance for numerical attributes
            numerical_distance = abs(row1[col] - row2[col])
            similarity += 1 - numerical_distance
            n_attributes += 1

        # Normalize similarity and express as percentage
        similarity_percentage = (similarity / n_attributes) * 100
        similarities.append(similarity_percentage)

    # Add similarity values to a new dataframe
    #similarity_df = pd.DataFrame({'Similarity (%)': similarities})
    final_similarity = sum(similarities)/len(similarities)
    print(final_similarity)
    return final_similarity