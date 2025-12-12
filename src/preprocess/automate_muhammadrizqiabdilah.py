import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    num_cols = ['Quantity', 'Price Per Unit', 'Total Spent']
    cat_cols = ['Item', 'Transaction Date']

    for feature in num_cols + cat_cols:
        df[feature] = df[feature].replace('ERROR', np.nan)
        df[feature] = df[feature].replace('UNKNOWN', np.nan)

    df[num_cols] = df[num_cols].astype(float)
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    df_selected = df[cat_cols + num_cols].copy()

    for col in num_cols:
        df_selected[col] = df_selected[col].fillna(df_selected[col].median())

    df_selected = df_selected.dropna()
    df_selected = df_selected.drop_duplicates()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_selected[num_cols])
    df_num_scaled = pd.DataFrame(scaled, columns=num_cols, index=df_selected.index)

    Q1 = df_num_scaled[num_cols].quantile(0.25)
    Q3 = df_num_scaled[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    mask = ~(
        (df_num_scaled[num_cols] < (Q1 - 1.5 * IQR)) |
        (df_num_scaled[num_cols] > (Q3 + 1.5 * IQR))
    ).any(axis=1)

    df_clean = pd.concat([
        df_num_scaled.loc[mask],
        df_selected.loc[mask, cat_cols]
    ], axis=1)

    df_clean["TotalSpent_Bin"] = pd.qcut(
        df_clean["Total Spent"],
        q=3, labels=["Low", "Medium", "High"]
    )

    return df_clean

