import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    num_cols = ['Quantity', 'Price Per Unit']   # target tidak disertakan
    target_col = 'Total Spent'
    cat_cols = ['Item', 'Transaction Date']

    # Replace error strings
    for feature in num_cols + cat_cols + [target_col]:
        df[feature] = df[feature].replace(['ERROR', 'UNKNOWN'], np.nan)

    # Convert types
    df[num_cols + [target_col]] = df[num_cols + [target_col]].astype(float)
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    # Select columns
    df_selected = df[num_cols + cat_cols + [target_col]].copy()

    # Fill missing
    for col in num_cols + [target_col]:
        df_selected[col] = df_selected[col].fillna(df_selected[col].median())

    # Drop other missing & duplicates
    df_selected = df_selected.dropna().drop_duplicates()

    # Scale only features, NOT target
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_selected[num_cols])
    df_num_scaled = pd.DataFrame(scaled, columns=num_cols, index=df_selected.index) 

    # Outlier filtering only on features
    Q1 = df_num_scaled[num_cols].quantile(0.25)
    Q3 = df_num_scaled[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~(
        (df_num_scaled[num_cols] < (Q1 - 1.5 * IQR)) |
        (df_num_scaled[num_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    # Merge cleaned features + target + categories
    df_clean = pd.concat([
        df_num_scaled.loc[mask],
        df_selected.loc[mask, cat_cols + [target_col]]
    ], axis=1)

    # Create bins for analytics (not used as feature)
    df_clean["TotalSpent_Bin"] = pd.qcut(
        df_clean[target_col],
        q=3, labels=["Low", "Medium", "High"]
    )

    # One hot encoding for Item
    encoder = OneHotEncoder(sparse_output=False)
    one_hot = encoder.fit_transform(df_clean[['Item']])
    one_hot_df = pd.DataFrame(
        one_hot,
        columns=encoder.get_feature_names_out(['Item']),
        index=df_clean.index
    )

    df_encoded = pd.concat([df_clean, one_hot_df], axis=1)
    df_encoded = df_encoded.drop('Item', axis=1)

    return df_encoded

def main():
    df = pd.read_csv("dirty_cafe_sales.csv")
    clean_df = preprocess_data(df)
    clean_df.to_csv("preprocessing/clean_dataset.csv", index=False)
    print("Preprocessing selesai. File disimpan di output/clean_dataset.csv")
    print(clean_df['Total Spent'].describe())

if __name__ == "__main__":
    main()
