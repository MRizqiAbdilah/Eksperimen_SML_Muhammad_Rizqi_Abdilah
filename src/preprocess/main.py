from .automate_muhammadrizqiabdilah import preprocess_data
import pandas as pd


def main():
    df = pd.read_csv("src/dirty_cafe_sales.csv")
    clean_df = preprocess_data(df)
    clean_df.to_csv("src/output/clean_dataset.csv", index=False)
    print("Preprocessing selesai. File disimpan di output/clean_dataset.csv")


if __name__ == "__main__":
    main()
